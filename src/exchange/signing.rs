//! Signing pipeline abstraction for exchange operations.
//!
//! This module provides a unified interface for signing Hyperliquid exchange actions.

use alloy::{
    primitives::{Address, Signature},
    signers::local::PrivateKeySigner,
};

use crate::{
    eip712::Eip712,
    helpers::next_nonce,
    prelude::*,
    serde_utils::hyperliquid_chain,
    signature::{sign_l1_action, sign_typed_data},
    ExchangeResponseStatus,
};

use super::exchange_client::{Actions, ExchangeClient};

/// Context for signing operations.
pub struct SigningContext<'a> {
    pub wallet: &'a PrivateKeySigner,
    pub vault_address: Option<Address>,
    pub is_mainnet: bool,
}

impl<'a> SigningContext<'a> {
    /// Create a new signing context.
    pub fn new(
        wallet: &'a PrivateKeySigner,
        vault_address: Option<Address>,
        is_mainnet: bool,
    ) -> Self {
        Self {
            wallet,
            vault_address,
            is_mainnet,
        }
    }

    /// Get the Hyperliquid chain name.
    pub fn chain_name(&self) -> &'static str {
        hyperliquid_chain(self.is_mainnet)
    }

    /// Get the chain name as an owned String (for struct fields).
    pub fn chain_name_string(&self) -> String {
        self.chain_name().to_string()
    }

    /// Sign an L1 action (MessagePack + keccak256 hash based signing).
    pub fn sign_l1_action(&self, action: &Actions, timestamp: u64) -> Result<Signature> {
        let connection_id = action.hash(timestamp, self.vault_address)?;
        sign_l1_action(self.wallet, connection_id, self.is_mainnet)
    }

    /// Sign a typed data action (EIP-712 based signing).
    pub fn sign_typed_data<T: Eip712>(&self, action: &T) -> Result<Signature> {
        sign_typed_data(action, self.wallet)
    }
}

impl ExchangeClient {
    /// Create a signing context for this client.
    pub fn signing_context<'a>(&'a self, wallet: Option<&'a PrivateKeySigner>) -> SigningContext<'a> {
        SigningContext::new(
            wallet.unwrap_or(&self.wallet),
            self.vault_address,
            self.http_client.is_mainnet(),
        )
    }

    /// Execute an L1 action with automatic signing and posting.
    ///
    /// This is the common pattern for most exchange operations:
    /// 1. Create action
    /// 2. Hash and sign
    /// 3. Post to exchange
    pub(crate) async fn execute_l1_action(
        &self,
        action: Actions,
        wallet: Option<&PrivateKeySigner>,
    ) -> Result<ExchangeResponseStatus> {
        let wallet = wallet.unwrap_or(&self.wallet);
        let timestamp = next_nonce();

        let connection_id = action.hash(timestamp, self.vault_address)?;
        let action_json =
            serde_json::to_value(&action).map_err(|e| crate::Error::JsonParse(e.to_string()))?;
        let is_mainnet = self.http_client.is_mainnet();
        let signature = sign_l1_action(wallet, connection_id, is_mainnet)?;

        self.post(action_json, signature, timestamp).await
    }

    /// Execute a typed data action with automatic signing and posting.
    ///
    /// This is used for EIP-712 signed actions like UsdSend, Withdraw3, etc.
    pub(crate) async fn execute_typed_data_action<T: Eip712 + serde::Serialize>(
        &self,
        action: T,
        action_wrapper: Actions,
        wallet: Option<&PrivateKeySigner>,
    ) -> Result<ExchangeResponseStatus> {
        let wallet = wallet.unwrap_or(&self.wallet);
        let timestamp = next_nonce();

        let signature = sign_typed_data(&action, wallet)?;
        let action_json = serde_json::to_value(action_wrapper)
            .map_err(|e| crate::Error::JsonParse(e.to_string()))?;

        self.post(action_json, signature, timestamp).await
    }
}
