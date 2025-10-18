#!/bin/bash

echo "🚀 BetYard ML Backend - Railway Deployment Setup"
echo "=" * 50

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
    echo "✅ Railway CLI installed!"
fi

# Login check
echo "🔐 Checking Railway authentication..."
if ! railway whoami &> /dev/null; then
    echo "🔑 Please login to Railway:"
    railway login
fi

# Initialize project
echo "📦 Initializing Railway project..."
railway init

# Deploy
echo "🚀 Deploying ML Backend to Railway..."
railway up

echo "=" * 50
echo "✅ Deployment Complete!"
echo "📡 Your ML Backend is now live on Railway"
echo "🔗 Check your Railway dashboard for the URL"
echo "💡 Update frontend ml-integration.js with your new URL"
echo "=" * 50