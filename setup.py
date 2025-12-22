from setuptools import setup, find_packages

setup(
    name="llm-data-wash",
    version="0.1.0",
    author="Your Name",
    author_email="your-email@example.com",
    description="A multi-GPU toolkit for regenerating and cleaning LLM training data",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/llm-data-wash",  # 替换为你的GitHub地址
    packages=find_packages(),
    install_requires=open("requirements.txt", encoding="utf-8").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires=">=3.8",
)