from setuptools import setup, find_packages

setup(
    name="tradeiq-backend",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0,<2.0.0",
        "sqlalchemy>=1.4.0,<2.0.0",
        "alembic>=1.7.0",
        "python-dotenv>=0.19.0",
    ],
    python_requires=">=3.8",
) 