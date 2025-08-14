# 🚀 Multi-Task LLM API

A powerful and elegant Flask API that provides text generation, code generation, and text classification using Google's **Gemini 2.0 Flash** model. Built with modern best practices including rate limiting, exponential backoff, comprehensive documentation, and interactive Swagger UI.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![Gemini](https://img.shields.io/badge/Gemini-2.0%20Flash-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

### 🎯 Core Functionality
- **📝 Text Generation** - Creative writing, summaries, and general text content
- **💻 Code Generation** - Programming assistance with optimized temperature settings
- **🏷️ Text Classification** - Categorize text into custom categories
- **🔄 Smart Rate Limiting** - Automatic handling of Gemini's 60 RPM limit with exponential backoff

### 🛡️ Enterprise Ready
- **📊 Interactive Swagger UI** - Complete API documentation and testing interface
- **⚡ Rate Limiting** - Flask-Limiter with memory storage (200/day, 50/hour, 10/minute per endpoint)
- **🔐 Secure Configuration** - Environment-based API key management
- **📈 Comprehensive Logging** - Detailed request/response tracking
- **🧪 Full Test Suite** - Unit tests covering all functionality including rate limiting

### 🏗️ Architecture Highlights
- **Modular Design** - Separation of concerns with dedicated wrapper layer
- **Error Handling** - Graceful error responses with proper HTTP status codes
- **Retry Logic** - Intelligent exponential backoff for API resilience
- **Documentation** - Auto-generated OpenAPI specification

## 🚦 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Cloud API key for Gemini
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Multi-Task_LLM_API
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

5. **Run the application**
```bash
python app.py
```

6. **Access the API**
- **Swagger UI**: http://0.0.0.0:8081/swagger/
- **API Base**: http://0.0.0.0:8081/api/v1/

## 📚 API Reference

### Base URL
```
http://0.0.0.0:8081/api/v1
```

### Endpoints

#### 📝 Text Generation
```http
POST /generate/text
Content-Type: application/json

{
  "prompt": "Write a short story about a robot"
}
```

**Response:**
```json
{
  "generated_text": "Unit 734, designated 'Custodian,' trundled down..."
}
```

#### 💻 Code Generation
```http
POST /generate/code
Content-Type: application/json

{
  "prompt": "Create a Python function to calculate fibonacci numbers"
}
```

**Response:**
```json
{
  "generated_code": "def fibonacci(n):\n    if n <= 1:\n        return n..."
}
```

#### 🏷️ Text Classification
```http
POST /classify/text
Content-Type: application/json

{
  "text": "This movie was amazing!",
  "categories": ["positive", "negative", "neutral"]
}
```

**Response:**
```json
{
  "classification": "positive"
}
```

### Rate Limits
- **Global**: 200 requests/day, 50 requests/hour
- **Per Endpoint**: 10 requests/minute
- **Gemini API**: 60 RPM with automatic retry handling

## 🔧 Configuration

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google Cloud API key for Gemini | ✅ Yes |

### Model Settings
- **Text Generation**: Gemini 2.0 Flash, Temperature: 0.7
- **Code Generation**: Gemini 2.0 Flash, Temperature: 0.2 (for consistency)
- **Text Classification**: Gemini 2.0 Flash, Temperature: 0.0 (for deterministic results)

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_unit.py
```

### Test Coverage
- ✅ Environment configuration validation
- ✅ All endpoint functionality testing
- ✅ Rate limiting and exponential backoff
- ✅ Error handling scenarios
- ✅ Direct wrapper function testing

## 📊 Rate Limiting & Retry Logic

### Exponential Backoff Strategy
- **Initial Delay**: 1 second
- **Max Delay**: 60 seconds
- **Max Retries**: 5 attempts
- **Multiplier**: Exponential (1s → 2s → 4s → 8s → 16s → 32s → 60s)

### Error Detection
The system automatically detects and handles:
- HTTP 429 (Too Many Requests)
- "Rate limit exceeded" messages
- "Quota exceeded" errors
- "Resource exhausted" responses

## 🏗️ Project Structure

```
Multi-Task_LLM_API/
├── app.py              # Flask application with Swagger integration
├── gemini_wrapper.py   # Gemini API wrapper with retry logic
├── requirements.txt    # Python dependencies
├── test_unit.py       # Comprehensive unit tests
├── .env               # Environment configuration
└── README.md          # Project documentation
```

## 📦 Dependencies

### Core Dependencies
```
Flask                 # Web framework
Flask-RESTX          # Swagger UI and API documentation
Flask-Limiter        # Rate limiting
python-dotenv        # Environment variable management
tenacity             # Retry and exponential backoff
```

### LLM Integration
```
langchain            # LLM framework
langchain-google-genai # Gemini integration
langchain-community  # Additional LangChain components
```

### Production
```
waitress             # WSGI server for production
```

## 🚀 Production Deployment

### Using Waitress (Recommended)
```bash
waitress-serve --host=0.0.0.0 --port=8081 app:app
```

### Alternative Production Setup
```bash
# Run with production settings
export FLASK_ENV=production
waitress-serve --host=0.0.0.0 --port=8081 --threads=4 app:app
```

## 🔍 Monitoring & Logging

The application provides comprehensive logging:

```python
# Example log output
INFO:gemini_wrapper:Generating text with Gemini 2.0 Flash
INFO:gemini_wrapper:Text generation completed successfully
WARNING:gemini_wrapper:Rate limit hit: 429 Too Many Requests. Retrying with exponential backoff...
```

## ⚠️ Error Handling

### HTTP Status Codes
- `200` - Success
- `400` - Bad Request (missing required fields)
- `429` - Too Many Requests (rate limit exceeded)
- `500` - Internal Server Error

### Error Response Format
```json
{
  "error": "Prompt is required"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini** for providing powerful AI capabilities
- **LangChain** for excellent LLM integration framework
- **Flask** ecosystem for robust web framework components

## 📞 Support

For support, questions, or feature requests:
- Create an issue in the repository
- Check the Swagger UI documentation at `/swagger/`
- Review the comprehensive test suite for usage examples

---

<div align="center">

**Built with ❤️ using Google Gemini 2.0 Flash**

[![Swagger](https://img.shields.io/badge/API_Docs-Swagger-green.svg)](http://0.0.0.0:8081/swagger/)
[![Tests](https://img.shields.io/badge/Tests-Comprehensive-blue.svg)](#-testing)
[![Rate_Limiting](https://img.shields.io/badge/Rate_Limiting-Smart-orange.svg)](#-rate-limiting--retry-logic)

</div>