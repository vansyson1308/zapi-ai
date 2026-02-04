"""
2api.ai Python SDK - Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="twoapi",
    version="1.0.0",
    author="2api.ai",
    author_email="support@2api.ai",
    description="Unified AI API - Access OpenAI, Anthropic, and Google through a single interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/2api-ai/twoapi-python",
    project_urls={
        "Documentation": "https://docs.2api.ai",
        "Bug Tracker": "https://github.com/2api-ai/twoapi-python/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Framework :: AsyncIO",
        "Typing :: Typed",
    ],
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "httpx>=0.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
        ],
    },
    keywords=[
        "ai",
        "openai",
        "anthropic",
        "claude",
        "gemini",
        "google",
        "gpt",
        "llm",
        "api",
        "sdk",
        "unified",
        "async",
        "tool-calling",
        "function-calling",
    ],
)
