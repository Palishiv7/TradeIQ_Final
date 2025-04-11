📊 Candlestick Pattern Assessment (Technical Analysis Training)

📌 Overview

The Candlestick Pattern Assessment is designed to test users on their ability to recognize and interpret candlestick patterns in financial markets. The assessment uses AI-powered image recognition to analyze candlestick patterns and provide real-time feedback.

🚀 Tech Stack

1️⃣ Backend (High-Performance API & AI Processing)

✅ Framework: FastAPI (Asynchronous API for speed and scalability)
✅ AI Model: ResNet-50 (Image recognition for candlestick pattern detection)
✅ Asynchronous Execution: Celery + Redis (For background AI processing)
✅ Authentication: JWT (Token-based authentication)
✅ WebSockets: FastAPI WebSockets (For real-time pattern validation)
✅ Database Connection: SQLAlchemy (PostgreSQL) + Motor (MongoDB for AI insights)

2️⃣ Frontend (Fast, Lightweight UI)

✅ Framework: React.js (Next.js for SSR & SEO optimization)
✅ UI Library: Tailwind CSS (For a fast and responsive UI)
✅ Charts & Graphs: Recharts.js (For candlestick visualization)
✅ Real-Time Features: WebSockets + React Query (For live market insights)
✅ State Management: Zustand (Simpler, faster than Redux)

3️⃣ Database (Hybrid Storage for Speed & Flexibility)

✅ Primary DB: PostgreSQL (Relational database for storing user progress, XP, scores)
✅ NoSQL DB: MongoDB (For storing AI-generated explanations & assessments)
✅ Cache & Session Storage: Redis (For instant response times & caching AI results)

📁 Project Directory Structure

tradeiq-assessments/
│── backend/
│   ├── assessments/
│   │   ├── candlestick_patterns/    # 📊 Technical Analysis Training
│   │   │   ├── candlestick_api.py   # API for Candlestick Assessment
│   │   │   ├── candlestick_ai.py    # ResNet-50 for Pattern Recognition
│   │   │   ├── candlestick_db.py    # Database Model
│   │   │   ├── candlestick_utils.py # Utility Functions
│   │   │   ├── candlestick_config.py# Configurations
│   ├── common/
│   │   ├── base_assessment.py       # Base class for all assessments
│   │   ├── database.py              # Shared DB Utilities
│   │   ├── cache.py                 # Redis Caching
│   │   ├── ai_engine.py             # AI Model Management
│   │   ├── config.py                # Global Configurations
│   ├── api.py                        # Unified API Gateway
│   ├── main.py                       # FastAPI Entry Point
│
│── frontend/
│   ├── components/
│   │   ├── CandlestickQuiz.jsx       # UI for Candlestick Assessment
│   ├── api/
│   │   ├── apiClient.js              # API Calls for Assessment
│   ├── pages/
│   │   ├── CandlestickPage.jsx       # Candlestick Quiz Page
│   ├── App.js                        # Main React App Entry
│
│── database/
│   ├── migrations/                    # Database Migrations
│   ├── init_db.py                      # Initialize Database
│
│── tests/
│   ├── test_candlestick.py             # Tests for Candlestick Assessment

🔗 How Components Communicate

1️⃣ User submits an image of a candlestick pattern via the UI (CandlestickQuiz.jsx).2️⃣ The frontend sends the image to the backend API (candlestick_api.py).3️⃣ The backend processes the image using ResNet-50 (candlestick_ai.py).4️⃣ The AI model classifies the pattern and sends the result back.5️⃣ The result is stored in PostgreSQL & MongoDB (candlestick_db.py).6️⃣ The frontend updates the user with feedback and XP points.

🧠 AI Model Used (ResNet-50)

Why ResNet-50?

Highly accurate image classification model.

Lightweight yet powerful for pattern recognition.

Pretrained on large-scale datasets for high performance.

How It Works?

Takes the user-submitted image and preprocesses it.

Passes the image through ResNet-50 for classification.

Returns the identified pattern (e.g., "Bullish Engulfing", "Doji").

📡 Backend API Endpoints

Method

Endpoint

Description

POST

/candlestick/upload

Uploads a candlestick image for AI analysis

GET

/candlestick/result/{user_id}

Retrieves user's latest assessment results

POST

/candlestick/submit-answer

Submits user’s answer and calculates score

GET

/candlestick/leaderboard

Fetches the leaderboard rankings

🚀 Next Steps

✅ Implement the backend API and integrate ResNet-50.

✅ Develop the frontend UI for users to upload images.

✅ Implement real-time leaderboard & progress tracking.

✅ Optimize AI inference using ONNX Runtime for performance.

This document provides a clear roadmap to follow for implementing the Candlestick Pattern Assessment. Once this is completed, we will move to the next assessment. 🚀

