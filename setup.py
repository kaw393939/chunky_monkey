# File: setup.py

from setuptools import setup, find_packages

setup(
    name="document_processor",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "aiofiles",
        "spacy",
        "tiktoken",
        "pydantic",
        "filelock",
        # Add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            'docprocess=document_processor.cli.commands:cli_main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A CLI tool for processing documents.",
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/yourusername/document_processor",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
