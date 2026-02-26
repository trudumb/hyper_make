use reqwest::{Client, Response};
use serde::Deserialize;
use std::time::Duration;
use tracing::warn;

use crate::{prelude::*, BaseUrl, Error};

/// HTTP status codes that indicate transient server errors (retryable)
const RETRYABLE_STATUS_CODES: &[u16] = &[502, 503, 504];

/// Maximum number of retry attempts for transient errors
const MAX_RETRIES: u32 = 3;

/// Initial backoff delay in milliseconds (doubles with each retry)
const INITIAL_BACKOFF_MS: u64 = 100;

#[derive(Deserialize, Debug)]
struct ErrorData {
    data: String,
    code: u16,
    msg: String,
}

#[derive(Debug)]
pub struct HttpClient {
    pub client: Client,
    pub base_url: String,
}

async fn parse_response(response: Response) -> Result<String> {
    let status_code = response.status().as_u16();
    let text = response
        .text()
        .await
        .map_err(|e| Error::GenericRequest(e.to_string()))?;

    if status_code < 400 {
        return Ok(text);
    }
    let error_data = serde_json::from_str::<ErrorData>(&text);
    if (400..500).contains(&status_code) {
        let client_error = match error_data {
            Ok(error_data) => Error::ClientRequest {
                status_code,
                error_code: Some(error_data.code),
                error_message: error_data.msg,
                error_data: Some(error_data.data),
            },
            Err(err) => Error::ClientRequest {
                status_code,
                error_message: text,
                error_code: None,
                error_data: Some(err.to_string()),
            },
        };
        return Err(client_error);
    }

    Err(Error::ServerRequest {
        status_code,
        error_message: text,
    })
}

impl HttpClient {
    /// Send a POST request with automatic retry for transient server errors (502, 503, 504).
    ///
    /// Uses exponential backoff: 100ms, 200ms, 400ms between retries.
    /// This handles transient errors from load balancers and server restarts.
    pub async fn post(&self, url_path: &'static str, data: String) -> Result<String> {
        let full_url = format!("{}{url_path}", self.base_url);

        for attempt in 0..=MAX_RETRIES {
            let request = self
                .client
                .post(&full_url)
                .header("Content-Type", "application/json")
                .body(data.clone())
                .build()
                .map_err(|e| Error::GenericRequest(e.to_string()))?;

            let result = self
                .client
                .execute(request)
                .await
                .map_err(|e| Error::GenericRequest(e.to_string()))?;

            let status = result.status().as_u16();

            // Check if this is a retryable error
            if RETRYABLE_STATUS_CODES.contains(&status) && attempt < MAX_RETRIES {
                let backoff = Duration::from_millis(INITIAL_BACKOFF_MS * 2u64.pow(attempt));
                warn!(
                    status = status,
                    attempt = attempt + 1,
                    max_attempts = MAX_RETRIES + 1,
                    backoff_ms = backoff.as_millis(),
                    url = %url_path,
                    "Retryable HTTP error, backing off"
                );
                tokio::time::sleep(backoff).await;
                continue;
            }

            return parse_response(result).await;
        }

        // This should never be reached due to the loop structure,
        // but return a clear error if it somehow is
        Err(Error::GenericRequest(format!(
            "Max retries ({MAX_RETRIES}) exceeded for {url_path}"
        )))
    }

    pub fn is_mainnet(&self) -> bool {
        self.base_url == BaseUrl::Mainnet.get_url()
    }
}
