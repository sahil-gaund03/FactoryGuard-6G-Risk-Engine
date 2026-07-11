# Streamlit Community Cloud Deployment Guide

This guide explains how to deploy **FactoryGuard 6G** on Streamlit Community Cloud.

## Deployment Steps

1. **Push code to GitHub:**
   Ensure your local repository is pushed to a public GitHub repository.
   ```bash
   git init
   git add .
   git commit -m "Initialize FactoryGuard 6G"
   git remote add origin https://github.com/yourusername/factoryguard-6g.git
   git branch -M main
   git push -u origin main
   ```

2. **Sign up / Log in to Streamlit Community Cloud:**
   Go to [share.streamlit.io](https://share.streamlit.io) and authorize using your GitHub account.

3. **Deploy the App:**
   * Click the **"New app"** button.
   * Select your repository (`factoryguard-6g`), branch (`main`), and set the main file path to:
     `app/Home.py`
   * Click **"Deploy!"**

4. **Verify App Execution:**
   Streamlit will spin up a server, install dependencies from `requirements.txt`, and boot the dashboard. Ensure all pages render, filters function, and data charts display correctly.
