use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConnectorError {
    #[error("WebSocket error: {0}")]
    WebSocketError(#[from] tokio_tungstenite::tungstenite::Error),
    
    #[error("JSON serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),
    
    #[error("Shared memory error: {0}")]
    SharedMemoryError(String),
    
    #[error("FlatBuffers error: {0}")]
    FlatBuffersError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("URL parse error: {0}")]
    UrlParseError(#[from] url::ParseError),
    
    #[error("Environment variable error: {0}")]
    EnvVarError(#[from] std::env::VarError),

    #[error("Timeout error: {0}")]
    TimeoutError(String),
}

pub type Result<T> = std::result::Result<T, ConnectorError>; 