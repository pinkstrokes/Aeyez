from pathlib import Path

from setuptools import find_packages, setup


def _read_requirements() -> list[str]:
    requirements_path = Path(__file__).parent / "requirements.txt"
    lines = requirements_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


setup(
    name="seeingeye",
    version="1.0.0",
    description="LangGraph rebuild of the SeeingEye multimodal reasoning pipeline",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=_read_requirements(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
