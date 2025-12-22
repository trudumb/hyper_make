//! Approval methods for ExchangeClient.
//!
//! This module contains methods for approving agents and builder fees.

use alloy::{
    primitives::{Address, B256},
    signers::local::PrivateKeySigner,
};

use crate::{
    exchange::actions::{ApproveAgent, ApproveBuilderFee},
    helpers::next_nonce,
    prelude::*,
    signature::sign_typed_data,
    Error, ExchangeResponseStatus,
};

use super::exchange_client::{Actions, ExchangeClient};

impl ExchangeClient {
    /// Approve an agent to trade on behalf of the user.
    ///
    /// Returns the agent's private key bytes and the exchange response.
    pub async fn approve_agent(
        &self,
        wallet: Option<&PrivateKeySigner>,
    ) -> Result<(B256, ExchangeResponseStatus)> {
        let wallet = wallet.unwrap_or(&self.wallet);
        let agent = PrivateKeySigner::random();

        let hyperliquid_chain = if self.http_client.is_mainnet() {
            "Mainnet".to_string()
        } else {
            "Testnet".to_string()
        };

        let nonce = next_nonce();
        let approve_agent = ApproveAgent {
            signature_chain_id: 421614,
            hyperliquid_chain,
            agent_address: agent.address(),
            agent_name: None,
            nonce,
        };
        let signature = sign_typed_data(&approve_agent, wallet)?;
        let action = serde_json::to_value(Actions::ApproveAgent(approve_agent))
            .map_err(|e| Error::JsonParse(e.to_string()))?;
        Ok((agent.to_bytes(), self.post(action, signature, nonce).await?))
    }

    /// Approve a builder fee for a specific builder address.
    pub async fn approve_builder_fee(
        &self,
        builder: Address,
        max_fee_rate: String,
        wallet: Option<&PrivateKeySigner>,
    ) -> Result<ExchangeResponseStatus> {
        let wallet = wallet.unwrap_or(&self.wallet);
        let timestamp = next_nonce();

        let hyperliquid_chain = if self.http_client.is_mainnet() {
            "Mainnet".to_string()
        } else {
            "Testnet".to_string()
        };

        let approve_builder_fee = ApproveBuilderFee {
            signature_chain_id: 421614,
            hyperliquid_chain,
            builder,
            max_fee_rate,
            nonce: timestamp,
        };
        let signature = sign_typed_data(&approve_builder_fee, wallet)?;
        let action = serde_json::to_value(Actions::ApproveBuilderFee(approve_builder_fee))
            .map_err(|e| Error::JsonParse(e.to_string()))?;

        self.post(action, signature, timestamp).await
    }
}
