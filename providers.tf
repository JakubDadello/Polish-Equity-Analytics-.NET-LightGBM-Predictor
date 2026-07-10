# Azure Provider source and version
terraform {
  required_providers {
    azurerm = {
        source  = "hashicorp/azurerm"
        version = "4.1.0"
    }
  }
}

# Microsoft Azure Provider configuration
provider "azurerm" {
  features {}
  use_cli = true

  # In version 4.0 of the azurerm provider it became mandatory to explicitly 
  # set the subscription_id in the provider block
  subscription_id = var.subscription_id
}