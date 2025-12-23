use thiserror::Error;

/// HTTP error classification
#[derive(Error, Debug, Clone)]
pub enum HttpErrorKind {
    #[error("Client error (code: {code:?}): {message}")]
    Client {
        code: Option<u16>,
        message: String,
        data: Option<String>,
    },
    #[error("Server error: {message}")]
    Server { message: String },
}

/// WebSocket-specific errors
#[derive(Error, Debug, Clone)]
pub enum WsError {
    #[error("Connection error: {0}")]
    Connection(String),
    #[error("Send error: {0}")]
    Send(String),
    #[error("Manager not instantiated")]
    ManagerNotFound,
    #[error("Subscription not found")]
    SubscriptionNotFound,
    #[error("Cannot subscribe to multiple user events")]
    MultipleUserEvents,
}

/// Signing and cryptographic errors
#[derive(Error, Debug, Clone)]
pub enum SigningError {
    #[error("EIP-712 error: {0}")]
    Eip712(String),
    #[error("ECDSA signature failed: {0}")]
    Ecdsa(String),
    #[error("Private key parse error: {0}")]
    PrivateKeyParse(String),
    #[error("Wallet error: {0}")]
    Wallet(String),
}

/// Parsing and serialization errors
#[derive(Error, Debug, Clone)]
pub enum ParseError {
    #[error("JSON error: {0}")]
    Json(String),
    #[error("MessagePack error: {0}")]
    Rmp(String),
    #[error("Invalid float string")]
    FloatString,
    #[error("Text conversion error: {0}")]
    TextConversion(String),
}

/// Main SDK error type
#[derive(Error, Debug, Clone)]
pub enum Error {
    // === New structured errors ===
    /// HTTP error with status code and classification
    #[error("HTTP error (status {status}): {kind}")]
    Http { status: u16, kind: HttpErrorKind },

    // === Legacy variants (for backward compatibility) ===
    /// Client HTTP error (4xx)
    #[error("Client error: status code: {status_code}, error code: {error_code:?}, error message: {error_message}, error data: {error_data:?}")]
    ClientRequest {
        status_code: u16,
        error_code: Option<u16>,
        error_message: String,
        error_data: Option<String>,
    },

    /// Server HTTP error (5xx)
    #[error("Server error: status code: {status_code}, error message: {error_message}")]
    ServerRequest {
        status_code: u16,
        error_message: String,
    },

    /// Generic request error
    #[error("Generic request error: {0}")]
    GenericRequest(String),

    /// Chain type not allowed
    #[error("Chain type not allowed for this function")]
    ChainNotAllowed,

    /// Asset not found
    #[error("Asset not found")]
    AssetNotFound,

    /// EIP-712 error
    #[error("Error from Eip712 struct: {0}")]
    Eip712(String),

    /// JSON parse error
    #[error("Json parse error: {0}")]
    JsonParse(String),

    /// Generic parse error
    #[error("Generic parse error: {0}")]
    GenericParse(String),

    /// Wallet error
    #[error("Wallet error: {0}")]
    Wallet(String),

    /// WebSocket connection error
    #[error("Websocket error: {0}")]
    Websocket(String),

    /// Subscription not found
    #[error("Subscription not found")]
    SubscriptionNotFound,

    /// WsManager not instantiated
    #[error("WS manager not instantiated")]
    WsManagerNotFound,

    /// WebSocket send error
    #[error("WS send error: {0}")]
    WsSend(String),

    /// Reader data not found
    #[error("Reader data not found")]
    ReaderDataNotFound,

    /// Generic reader error
    #[error("Reader error: {0}")]
    GenericReader(String),

    /// Reader text conversion error
    #[error("Reader text conversion error: {0}")]
    ReaderTextConversion(String),

    /// Order type not found
    #[error("Order type not found")]
    OrderTypeNotFound,

    /// Random generation error
    #[error("Issue with generating random data: {0}")]
    RandGen(String),

    /// Private key parse error
    #[error("Private key parse error: {0}")]
    PrivateKeyParse(String),

    /// Multiple user events subscription error
    #[error("Cannot subscribe to multiple user events")]
    UserEvents,

    /// MessagePack parse error
    #[error("Rmp parse error: {0}")]
    RmpParse(String),

    /// Float string parse error
    #[error("Invalid input number")]
    FloatStringParse,

    /// No cloid found
    #[error("No cloid found in order request when expected")]
    NoCloid,

    /// Signature failure
    #[error("ECDSA signature failed: {0}")]
    SignatureFailure(String),

    /// Vault address not found
    #[error("Vault address not found")]
    VaultAddressNotFound,
}

// Convenience constructors for common error patterns
impl Error {
    /// Create an HTTP client error
    pub fn client_error(
        status: u16,
        code: Option<u16>,
        message: String,
        data: Option<String>,
    ) -> Self {
        Error::Http {
            status,
            kind: HttpErrorKind::Client {
                code,
                message,
                data,
            },
        }
    }

    /// Create an HTTP server error
    pub fn server_error(status: u16, message: String) -> Self {
        Error::Http {
            status,
            kind: HttpErrorKind::Server { message },
        }
    }

    /// Create a JSON parse error (uses new structured type)
    pub fn json_parse(msg: impl Into<String>) -> Self {
        Error::JsonParse(msg.into())
    }

    /// Create a WebSocket connection error
    pub fn ws_connection(msg: impl Into<String>) -> Self {
        Error::Websocket(msg.into())
    }

    /// Create a signing/EIP-712 error
    pub fn eip712(msg: impl Into<String>) -> Self {
        Error::Eip712(msg.into())
    }

    /// Create a signature failure error
    pub fn signature_failure(msg: impl Into<String>) -> Self {
        Error::SignatureFailure(msg.into())
    }
}
