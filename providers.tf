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
}