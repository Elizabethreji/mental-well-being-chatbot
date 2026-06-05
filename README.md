# 🧠 AI-Powered Mental Well-Being Chatbot

An intelligent mental health support chatbot that combines Large Language Models (LLMs), Facial Emotion Recognition, Speech-to-Text technology, and Vector Search to provide personalized and empathetic mental well-being assistance.

## 📌 Overview

Mental health support is often limited by stigma, accessibility issues, and the availability of professionals. This project aims to bridge that gap by providing a secure, accessible, and AI-driven platform where users can express their thoughts and emotions through text, voice, and facial expressions.

The chatbot leverages advanced AI technologies to understand user emotions, maintain contextual conversations, and provide supportive responses in real time.

## ✨ Features

- 💬 AI-powered conversational chatbot
- 🎙️ Speech-to-Text using Whisper ASR
- 😊 Real-time Facial Emotion Detection
- 🧠 Context-aware response generation using LLaMA
- 🔍 Vector-based memory retrieval using ChromaDB
- 🔐 Secure user authentication
- 📜 Chat history storage
- 🌿 Personalized mental wellness recommendations
- ☁️ Scalable cloud deployment support

## 🏗️ System Architecture

### Core Components

#### 1. Natural Language Processing
- LLaMA Transformer Model
- LangChain Integration
- Context-aware response generation

#### 2. Speech Recognition
- OpenAI Whisper ASR
- Converts voice input into text

#### 3. Facial Emotion Recognition
- DeepFace
- OpenCV
- TensorFlow
- Real-time emotion analysis

#### 4. Vector Database
- ChromaDB
- Semantic search for previous conversations

#### 5. Authentication & Security
- Firebase Authentication
- SHA-256 Hashing
- Bcrypt Password Encryption

#### 6. Database
- Firebase Firestore
- ChromaDB

## 🛠️ Technology Stack

| Category | Technologies |
|-----------|-------------|
| Frontend | Gradio |
| Backend | Python |
| LLM | LLaMA Transformer |
| Framework | LangChain |
| Speech Recognition | OpenAI Whisper |
| Emotion Detection | DeepFace, OpenCV, TensorFlow |
| Database | Firebase Firestore, ChromaDB |
| Authentication | Firebase Auth, SHA-256, Bcrypt |
| Deployment | AWS / Azure / GCP |

## 🚀 Workflow

1. User enters text, voice, or video input.
2. Whisper transcribes audio into text.
3. Facial emotion detection analyzes emotional state.
4. ChromaDB retrieves relevant conversation context.
5. LLaMA generates a personalized response.
6. Response is displayed to the user.
7. Chat history and embeddings are stored for future interactions.


## 🎯 Objectives

- Provide accessible mental health support.
- Understand emotions through text and facial cues.
- Generate empathetic and context-aware responses.
- Promote mindfulness and emotional well-being.
- Ensure user privacy and security.

## 🔒 Privacy & Security

- Secure authentication using Firebase.
- Password encryption using Bcrypt and SHA-256.
- Safe storage of user conversations.
- Privacy-focused design for mental health interactions.


## 📸 Screenshots

### Login Page
<img src="screenshots/login.png" width="700">

### Sign Up Page
<img src="screenshots/signup.png" width="700">

### Mental Well-Being Chatbot
<img src="screenshots/chatbot.png" width="700">

## 👩‍💻 Authors

- Elizabeth Reji
- Team Members

## 📚 References

1. Yuan et al. (2024) - Improving Workplace Well-being in Modern Organizations.
2. Wang & Farb (2024) - Chatbot-Based Interventions for Mental Health Support.
3. Abilkaiyrkyzy et al. (2024) - Dialogue System for Early Mental Illness Detection.
4. Mehta et al. (2022) - AI Powered Chatbot for Mental Healthcare based on Sentiment Analysis.

## ⭐ Acknowledgements

This project was developed as part of a B.Tech Computer Science and Engineering Mini Project focused on leveraging Artificial Intelligence for Mental Well-Being Support.
