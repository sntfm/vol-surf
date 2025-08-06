mod websocket;
mod shared_mem;
mod error;
mod models;

use dotenv::dotenv;
use log::info;
use std::env;
use tokio;

use crate::websocket::PolygonWebSocket;
use crate::error::Result;

const POLYGON_WS_URL: &str = "wss://delayed.polygon.io/options";

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize environment
    dotenv().ok();
    env_logger::init();
    
    // Get API key from environment
    let api_key = env::var("POLYGON_API_KEY")
        .expect("POLYGON_API_KEY must be set in environment");
        
    info!("Starting Polygon.io options data connector");
    
    // Initialize WebSocket client
    let mut ws_client = PolygonWebSocket::new(POLYGON_WS_URL, &api_key)?;
    
    // Start the WebSocket connection and processing loop
    ws_client.connect_and_process().await?;
    
    Ok(())
}
