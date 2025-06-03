# Stock Market Prediction using Machine Learning

A comprehensive machine learning system for predicting stock market movements of Indian companies using historical data and advanced ML techniques.

## Features

- Historical stock data analysis from 2002 to 2021
- Multiple ML models including LSTM, Random Forest, and XGBoost
- Real-time prediction API
- Interactive dashboard for visualization
- Automated data pipeline
- Comprehensive testing suite
- Production-ready deployment configurations

## Project Structure

```
stock_prediction/
├── data/                  # Data files and data processing scripts
├── models/               # ML model implementations
├── api/                  # FastAPI implementation
├── dashboard/           # Dash/Plotly dashboard
├── tests/               # Test suite
├── config/              # Configuration files
├── notebooks/           # Jupyter notebooks for analysis
└── utils/               # Utility functions
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-prediction-ml.git
cd stock-prediction-ml
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models

```bash
python scripts/train_models.py --config config/training_config.yaml
```

### Running the API

```bash
uvicorn api.main:app --reload
```

### Running the Dashboard

```bash
python dashboard/app.py
```

### Running Tests

```bash
pytest tests/
```

## Data Pipeline

The data pipeline includes:
1. Data collection from various sources
2. Data preprocessing and cleaning
3. Feature engineering
4. Model training and validation
5. Performance monitoring

## Models

The project implements several models:
- LSTM for sequence prediction
- Random Forest for classification
- XGBoost for regression
- Ensemble methods for combining predictions

## API Documentation

The API is documented using OpenAPI (Swagger) and can be accessed at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Dashboard

The dashboard provides:
- Real-time stock predictions
- Historical performance analysis
- Model comparison metrics
- Custom stock analysis tools

## Testing

The project includes:
- Unit tests
- Integration tests
- End-to-end tests
- Performance tests

## Deployment

Deployment configurations are provided for:
- Docker
- Kubernetes
- Cloud platforms (AWS, GCP, Azure)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Security

- API authentication using JWT
- Environment variable management
- Secure data handling
- Regular security updates

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NSE India for providing historical data
- Contributors and maintainers
- Open source community
