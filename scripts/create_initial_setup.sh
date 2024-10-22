#!bin/bash

az group create -n customer0123tfstate -l eastus2
 
az storage account create -n customer0123tfstate -g customer0123tfstate -l eastus2 --sku "Standard_LRS"
 
az storage container create -n tfstate --account-name customer0123tfstate