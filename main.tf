###############################################################
# Terraform IaC – Azure Infrastructure
# 
# Overview:
#   Main Terraform configuration file responsible for provisioning
#   core Azure resources required for the ML/AI inference application.
#
# This configuration defines:
#   - Resource Group
#   - Container Registry
#   - Container Apps Environment
#   - Container App (API inference service)
#
# Architecture Notes:
#   - Current deployment: Azure Container Apps (lightweight, serverless)
#   - Future scalability (hypothetical): AKS with GPU-enabled node pools
#
# Purpose:
#   - Provide reproducible, version-controlled infrastructure
#   - Enable automated provisioning and clean environment setup
#   - Establish a foundation that can scale into a production-grade architecture
###############################################################

# Resource group
resource "azurerm_resource_group" "psq" {
  name     = "psq-stock-exchange-equity-analitycs"
  location = "Poland Central"
}

# Azure Blob Storage (S3 equivalent)
resource "azurerm_storage_account" "storage" {
    name  = "blob1stock1exchange1ea"
    resource_group_name = azurerm_resource_group.psq.name #related to Resource Group by the name
    location = azurerm_resource_group.psq.location #related to Resource Group by the location
    account_tier = "Standard"
    account_replication_type = "LRS" #Locally Redundant Storage  
}


resource "azurerm_storage_container" "models" {
  name                  = "models"
  storage_account_name = azurerm_storage_account.storage.name
  container_access_type = "private"
}

# Azure Container Registry (ECR equivalent)
resource "azurerm_container_registry" "acr" {
  name                = "acr1stock1exchange1equity1analitycs"
  resource_group_name = azurerm_resource_group.psq.name
  location            = azurerm_resource_group.psq.location
  sku                 = "Standard"  #with no georeplications
  admin_enabled       = false  #set 
}

# Azure User Assigned Identity (IAM equivalent )
resource "azurerm_user_assigned_identity" "uai" {
  name                = "aci-uai"
  resource_group_name = azurerm_resource_group.psq.name
  location            = azurerm_resource_group.psq.location
}

# Azure UAI Role Assignment (IAM Role equivalent )
resource "azurerm_role_assignment" "aci_acr_pull" {
  principal_id         = azurerm_user_assigned_identity.uai.principal_id
  role_definition_name = "AcrPull"
  scope = azurerm_container_registry.acr.id
}

# Azure Container Instances (ECS equivalent)
resource "azurerm_container_group" "aci" {
  name                = "aci-inference-app"
  resource_group_name = azurerm_resource_group.psq.name
  location            = azurerm_resource_group.psq.location
  os_type             = "Linux"
  ip_address_type     = "Public"
  dns_name_label      = "stock-exchange-ea-inference" #public URL endpoint

  identity {
    type = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.uai.id]
  }

  container {
    name   = "inference-api"
    image = "${azurerm_container_registry.acr.login_server}/azure-inference-api:latest"
    cpu    = "1.0"
    memory = "1.5"

    ports {
      port     = 8080  
      protocol = "TCP"  #transport protocol = TCP
    }

    environment_variables = {
      "StorageConnectionString" = azurerm_storage_account.storage.primary_connection_string
    }
  }
}

# Azure Log Analytics Workspace (CloudWatch equivalent)
resource "azurerm_log_analytics_workspace" "logs" {
  name                = "logs1stock1exchange1equity1analitycs"
  location            = azurerm_resource_group.psq.location
  resource_group_name = azurerm_resource_group.psq.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

# Diagnostic Settings for ACI (send logs and metrics to resource "logs")
resource "azurerm_monitor_diagnostic_setting" "aci_diagnostics" {
  name                       = "aci-diagnostics"
  target_resource_id         = azurerm_container_group.aci.id
  log_analytics_workspace_id = azurerm_log_analytics_workspace.logs.id

  # Metrics
  metric {
    category = "AllMetrics"
    enabled  = true
  }
}

# Azure Storage Management Policy
resource "azurerm_storage_management_policy" "retention" {
  storage_account_id = azurerm_storage_account.storage.id

  rule {
    name    = "delete-old-logs"
    enabled = true

    filters {
      prefix_match = ["logs"]
      blob_types   = ["blockBlob"]
    }

    actions {
      base_blob {
        delete_after_days_since_modification_greater_than = 30
      }
    }
  }
}
