# HousingRegression

For this I have used Docker functionality to execute it locally without any system compatibility issue.

## Docker Usage

### Pre-Requistes

Colima/Docker

To build and run the project using Docker:

```bash
docker-compose up --build
```

Above command will not the save the plot graph in the local to do so run below

```bash
docker run -v $(pwd)/output:/app/output your_image_name
```

## Directory Structure

```
HousingRegression/
|-- .github/workflows/
|   |-- ci.yml
|-- utils.py
|-- regression.py
|-- requirements.txt
|-- README.md
```