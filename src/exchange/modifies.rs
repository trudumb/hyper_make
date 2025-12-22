//! Order modification methods for ExchangeClient.
//!
//! This module contains methods for modifying existing orders.

use alloy::signers::local::PrivateKeySigner;

use crate::{
    exchange::{
        actions::BulkModify,
        modify::{ClientModifyRequest, ModifyRequest},
    },
    helpers::next_nonce,
    prelude::*,
    signature::sign_l1_action,
    Error, ExchangeResponseStatus,
};

use super::exchange_client::{Actions, ExchangeClient};

impl ExchangeClient {
    /// Modify a single order.
    pub async fn modify(
        &self,
        modify: ClientModifyRequest,
        wallet: Option<&PrivateKeySigner>,
    ) -> Result<ExchangeResponseStatus> {
        self.bulk_modify(vec![modify], wallet).await
    }

    /// Modify multiple orders in a single request.
    pub async fn bulk_modify(
        &self,
        modifies: Vec<ClientModifyRequest>,
        wallet: Option<&PrivateKeySigner>,
    ) -> Result<ExchangeResponseStatus> {
        let wallet = wallet.unwrap_or(&self.wallet);
        let timestamp = next_nonce();

        let mut transformed_modifies = Vec::new();
        for modify in modifies.into_iter() {
            transformed_modifies.push(ModifyRequest {
                oid: modify.oid,
                order: modify.order.convert(&self.coin_to_asset)?,
            });
        }

        let action = Actions::BatchModify(BulkModify {
            modifies: transformed_modifies,
        });
        let connection_id = action.hash(timestamp, self.vault_address)?;

        let action = serde_json::to_value(&action).map_err(|e| Error::JsonParse(e.to_string()))?;
        let is_mainnet = self.http_client.is_mainnet();
        let signature = sign_l1_action(wallet, connection_id, is_mainnet)?;

        self.post(action, signature, timestamp).await
    }
}
