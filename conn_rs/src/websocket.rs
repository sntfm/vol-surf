use futures_util::{SinkExt, StreamExt};
use log::{info, error, debug, warn};
use serde::{Deserialize, Serialize};
use tokio_tungstenite::{
    connect_async,
    tungstenite::protocol::Message,
};
use tokio::time::{timeout, Duration};
use url::Url;
use serde_json::Value;

use crate::error::{Result, ConnectorError};
use crate::shared_mem::SharedMemManager;

const CONNECTION_TIMEOUT: Duration = Duration::from_secs(10);
const SUBSCRIPTION_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Debug, Serialize)]
struct SubscriptionMessage {
    action: String,
    params: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum PolygonMessage {
    Status {
        #[serde(rename = "ev")]
        event: String,
        status: String,
        message: Option<String>,
    },
    Data {
        #[serde(rename = "ev")]
        event: String,
        #[serde(flatten)]
        data: Value,
    },
    Unknown(Value),
}

pub struct PolygonWebSocket {
    url: String,
    api_key: String,
    shared_mem: SharedMemManager,
}

impl PolygonWebSocket {
    pub fn new(base_url: &str, api_key: &str) -> Result<Self> {
        Ok(Self {
            url: format!("{}?apiKey={}", base_url, api_key),
            api_key: api_key.to_string(),
            shared_mem: SharedMemManager::new()?,
        })
    }
    
    pub async fn connect_and_process(&mut self) -> Result<()> {
        let url = Url::parse(&self.url)
            .map_err(ConnectorError::UrlParseError)?;
            
        // Add timeout for initial connection
        let ws_stream = match timeout(CONNECTION_TIMEOUT, connect_async(url)).await {
            Ok(Ok((stream, _))) => stream,
            Ok(Err(e)) => return Err(ConnectorError::WebSocketError(e)),
            Err(_) => return Err(ConnectorError::TimeoutError("Connection timeout".to_string())),
        };
            
        info!("WebSocket connection established");
        
        let (mut write, mut read) = ws_stream.split();
        
        // Subscribe to all options
        let subscribe_msg = SubscriptionMessage {
            action: "subscribe".to_string(),
            params: "O.*".to_string(),
        };
        
        let subscribe_json = serde_json::to_string(&subscribe_msg)
            .map_err(ConnectorError::SerdeError)?;
            
        // Send subscription request with timeout
        match timeout(CONNECTION_TIMEOUT, write.send(Message::Text(subscribe_json))).await {
            Ok(Ok(_)) => info!("Subscription request sent"),
            Ok(Err(e)) => return Err(ConnectorError::WebSocketError(e)),
            Err(_) => return Err(ConnectorError::TimeoutError("Subscription request timeout".to_string())),
        }
        
        let mut authenticated = false;
        let mut subscription_timer = tokio::time::Instant::now();
        
        // Process incoming messages
        while let Some(msg) = read.next().await {
            // Check subscription timeout
            if !authenticated && subscription_timer.elapsed() > SUBSCRIPTION_TIMEOUT {
                return Err(ConnectorError::TimeoutError(
                    "Timed out waiting for subscription confirmation".to_string()
                ));
            }

            match msg {
                Ok(Message::Text(text)) => {
                    debug!("Received message: {}", text);
                    
                    let polygon_msg: PolygonMessage = match serde_json::from_str(&text) {
                        Ok(msg) => msg,
                        Err(e) => {
                            error!("Failed to parse message: {}", e);
                            error!("Raw message: {}", text);
                            continue;
                        }
                    };
                    
                    match polygon_msg {
                        PolygonMessage::Status { event, status, message } => {
                            match event.as_str() {
                                "status" => {
                                    match status.as_str() {
                                        "connected" => {
                                            info!("Successfully connected to Polygon.io");
                                            authenticated = true;
                                        },
                                        "auth_success" => {
                                            info!("Successfully authenticated with Polygon.io");
                                            authenticated = true;
                                        },
                                        "auth_failed" => {
                                            let msg = message.clone().unwrap_or_default();
                                            error!("Authentication failed: {}", msg);
                                            return Err(ConnectorError::SharedMemoryError(
                                                format!("Authentication failed: {}", msg)
                                            ));
                                        },
                                        "error" => {
                                            let msg = message.clone().unwrap_or_default();
                                            error!("Received error from Polygon.io: {}", msg);
                                            if msg.contains("subscription") {
                                                warn!("Subscription error - this might be due to account tier limitations");
                                                warn!("Please check your Polygon.io subscription plan");
                                                return Err(ConnectorError::SharedMemoryError(
                                                    format!("Subscription error: {}", msg)
                                                ));
                                            }
                                        },
                                        _ => {
                                            warn!("Unknown status received: {} - {}", 
                                                status, 
                                                message.unwrap_or_default()
                                            );
                                        }
                                    }
                                },
                                _ => {
                                    debug!("Unhandled event type: {}", event);
                                }
                            }
                        },
                        PolygonMessage::Data { data, .. } => {
                            if !authenticated {
                                warn!("Received data before authentication confirmation");
                                continue;
                            }
                            
                            // Process the message and update shared memory
                            if let Err(e) = self.process_message(data).await {
                                error!("Failed to process message: {}", e);
                            }
                        },
                        PolygonMessage::Unknown(value) => {
                            debug!("Received unknown message format: {:?}", value);
                        }
                    }
                }
                Ok(Message::Close(frame)) => {
                    info!("WebSocket connection closed: {:?}", frame);
                    break;
                }
                Err(e) => {
                    error!("WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }
        
        Ok(())
    }
    
    async fn process_message(&mut self, data: Value) -> Result<()> {
        // Update shared memory with the new data
        self.shared_mem.update_data(&data)?;
        Ok(())
    }
} 