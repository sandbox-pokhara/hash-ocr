import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hash-ocr",
    version="1.0.1",
    author="Pradish Bijukchhe",
    author_email="pradishbijukchhe@gmail.com",
    description="An ocr designed to read in game texts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sandbox-pokhara/hash-ocr",
    project_urls={
        "Bug Tracker": "https://github.com/sandbox-pokhara/hash-ocr/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True,
    package_dir={"hash_ocr": "hash_ocr"},
    python_requires=">=3",
    install_requires=["opencv-python", "opencv-contrib-python"],
)
