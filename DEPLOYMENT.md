# Vercel Deployment Instructions

## Setup

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Set Environment Variables**:
   Edit `.env` file and set your keys:
   - `AUTH_KEY`: Your authentication key for login
   - `SECRET_KEY`: Flask session secret key

3. **Deploy to Vercel**:
   ```bash
   vercel
   ```

4. **Set Environment Variables on Vercel**:
   ```bash
   vercel env add AUTH_KEY
   vercel env add SECRET_KEY
   ```

## Local Testing

1. **Set environment variables**:
   Edit `.env` file with your keys

2. **Run locally**:
   ```bash
   python app.py
   ```

3. **Access**:
   - Open http://localhost:5001
   - Login with your AUTH_KEY

## Features

- ✅ Login authentication required
- ✅ Separate pest and egg counting
- ✅ Red circles = Active pests (>=150px)
- ✅ Orange circles = Eggs (<150px)
- ✅ Leaf structure analysis
- ✅ Shadow removal from borders

## Default Keys (Change These!)

- AUTH_KEY: `your-secret-key-here`
- SECRET_KEY: `your-flask-secret-key-here`
