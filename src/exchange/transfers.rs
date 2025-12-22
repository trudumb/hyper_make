//! Transfer methods for ExchangeClient.
//!
//! This module contains methods for various transfer operations:
//! - USDC transfers
//! - Asset transfers (send_asset)
//! - Spot transfers
//! - Vault transfers
//! - Bridge withdrawals
//! - Class transfers (perp <-> spot)

use alloy::signers::local::PrivateKeySigner;

use crate::{
    exchange::actions::{SendAsset, UsdSend},
    helpers::next_nonce,
    prelude::*,
    signature::{sign_l1_action, sign_typed_data},
    ClassTransfer, Error, ExchangeResponseStatus, SpotSend, SpotUser, VaultTransfer, Withdraw3,
};

use super::exchange_client::{Actions, ExchangeClient};

impl ExchangeClient {
    /// Transfer USDC to another address.
    pub async fn usdc_transfer(
        &self,
        amount: &str,
        destination: &str,
        wallet: Option<&PrivateKeySigner>,
    ) -> Result<ExchangeResponseStatus> {
        let wallet = wallet.unwrap_or(&self.wallet);
        let hyperliquid_chain = if self.http_client.is_mainnet() {
            "Mainnet".to_string()
        } else {
            "Testnet".to_string()
        };

        let timestamp = next_nonce();
        let usd_send = UsdSend {
            signature_chain_id: 421614,
            hyperliquid_chain,
            destination: destination.to_string(),
            amount: amount.to_string(),
            time: timestamp,
        };
        let signature = sign_typed_data(&usd_send, wallet)?;
        let action = serde_json::to_value(Actions::UsdSend(usd_send))
            .map_err(|e| Error::JsonParse(e.to_string()))?;

        self.post(action, signature, timestamp).await
    }

    /// Transfer funds between perp and spot accounts.
    pub async fn class_transfer(
        &self,
        usdc: f64,
        to_perp: bool,
        wallet: Option<&PrivateKeySigner>,
    ) -> Result<ExchangeResponseStatus> {
        // payload expects usdc without decimals
        let usdc = (usdc * 1e6).round() as u64;
        let wallet = wallet.unwrap_or(&self.wallet);

        let timestamp = next_nonce();

        let action = Actions::SpotUser(SpotUser {
            class_transfer: ClassTransfer { usdc, to_perp },
        });
        let connection_id = action.hash(timestamp, self.vault_address)?;
        let action = serde_json::to_value(&action).map_err(|e| Error::JsonParse(e.to_string()))?;
        let is_mainnet = self.http_client.is_mainnet();
        let signature = sign_l1_action(wallet, connection_id, is_mainnet)?;

        self.post(action, signature, timestamp).await
    }

    /// Send an asset to another address.
    pub async fn send_asset(
        &self,
        destination: &str,
        source_dex: &str,
        destination_dex: &str,
        token: &str,
        amount: f64,
        wallet: Option<&PrivateKeySigner>,
    ) -> Result<ExchangeResponseStatus> {
        let wallet = wallet.unwrap_or(&self.wallet);

        let hyperliquid_chain = if self.http_client.is_mainnet() {
            "Mainnet".to_string()
        } else {
            "Testnet".to_string()
        };

        let timestamp = next_nonce();

        // Build fromSubAccount string (similar to Python SDK)
        let from_sub_account = self
            .vault_address
            .map_or_else(String::new, |vault_addr| format!("{vault_addr:?}"));

        let send_asset = SendAsset {
            signature_chain_id: 421614,
            hyperliquid_chain,
            destination: destination.to_string(),
            source_dex: source_dex.to_string(),
            destination_dex: destination_dex.to_string(),
            token: token.to_string(),
            amount: amount.to_string(),
            from_sub_account,
            nonce: timestamp,
        };

        let signature = sign_typed_data(&send_asset, wallet)?;
        let action = serde_json::to_value(Actions::SendAsset(send_asset))
            .map_err(|e| Error::JsonParse(e.to_string()))?;

        self.post(action, signature, timestamp).await
    }

    /// Transfer to/from a vault.
    pub async fn vault_transfer(
        &self,
        is_deposit: bool,
        usd: u64,
        vault_address: Option<alloy::primitives::Address>,
        wallet: Option<&PrivateKeySigner>,
    ) -> Result<ExchangeResponseStatus> {
        let vault_address = self
            .vault_address
            .or(vault_address)
            .ok_or(Error::VaultAddressNotFound)?;
        let wallet = wallet.unwrap_or(&self.wallet);

        let timestamp = next_nonce();

        let action = Actions::VaultTransfer(VaultTransfer {
            vault_address,
            is_deposit,
            usd,
        });
        let connection_id = action.hash(timestamp, self.vault_address)?;
        let action = serde_json::to_value(&action).map_err(|e| Error::JsonParse(e.to_string()))?;
        let is_mainnet = self.http_client.is_mainnet();
        let signature = sign_l1_action(wallet, connection_id, is_mainnet)?;

        self.post(action, signature, timestamp).await
    }

    /// Withdraw funds from the bridge.
    pub async fn withdraw_from_bridge(
        &self,
        amount: &str,
        destination: &str,
        wallet: Option<&PrivateKeySigner>,
    ) -> Result<ExchangeResponseStatus> {
        let wallet = wallet.unwrap_or(&self.wallet);
        let hyperliquid_chain = if self.http_client.is_mainnet() {
            "Mainnet".to_string()
        } else {
            "Testnet".to_string()
        };

        let timestamp = next_nonce();
        let withdraw = Withdraw3 {
            signature_chain_id: 421614,
            hyperliquid_chain,
            destination: destination.to_string(),
            amount: amount.to_string(),
            time: timestamp,
        };
        let signature = sign_typed_data(&withdraw, wallet)?;
        let action = serde_json::to_value(Actions::Withdraw3(withdraw))
            .map_err(|e| Error::JsonParse(e.to_string()))?;

        self.post(action, signature, timestamp).await
    }

    /// Transfer spot tokens to another address.
    pub async fn spot_transfer(
        &self,
        amount: &str,
        destination: &str,
        token: &str,
        wallet: Option<&PrivateKeySigner>,
    ) -> Result<ExchangeResponseStatus> {
        let wallet = wallet.unwrap_or(&self.wallet);
        let hyperliquid_chain = if self.http_client.is_mainnet() {
            "Mainnet".to_string()
        } else {
            "Testnet".to_string()
        };

        let timestamp = next_nonce();
        let spot_send = SpotSend {
            signature_chain_id: 421614,
            hyperliquid_chain,
            destination: destination.to_string(),
            amount: amount.to_string(),
            time: timestamp,
            token: token.to_string(),
        };
        let signature = sign_typed_data(&spot_send, wallet)?;
        let action = serde_json::to_value(Actions::SpotSend(spot_send))
            .map_err(|e| Error::JsonParse(e.to_string()))?;

        self.post(action, signature, timestamp).await
    }
}
