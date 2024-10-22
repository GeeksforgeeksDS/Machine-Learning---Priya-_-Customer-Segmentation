terraform {
  backend "azurerm" {
    resource_group_name  = "customer0123tfstate"
    storage_account_name = "customer0123tfstate"
    container_name       = "tfstate"
    key                  = "prod.terraform.tfstate"
  }
}

provider "azurerm" {
  features {}
}

module "web_app" {
  source = "./web_app"
}
