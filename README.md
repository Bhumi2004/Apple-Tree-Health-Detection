# AI-Powered Apple Tree Health Detection

This project aims to provide a comprehensive solution for monitoring the health of apple trees using AI. It is divided into two main components:

1. **Apple Detection**: Detects whether apples are ripe, unripe, or overripe, and counts the total number of apples.
2. **Apple Leaf Detection**: Analyzes the health of apple leaves to identify if they are healthy or affected by scab disease.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Apple Detection](#apple-ripeness-detection)
  - [Apple Leaf Detection](#apple-leaf-health-detection)
- [Demo](#demo)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

```
├── Apple-Detection/
│   ├── apple.py
│   ├── kaggle.json
│   ├── leaf.py
|   ├── main.py
|   ├── styled_description.html
│   └── yolov10n.pt
   
|   
├── Apple-Leaf-Detection/
│   ├── leafdetect
|   |  └── helathy_apple
|   |  └── scaby_apple
│   ├── leafclassification_model.keras
│   ├── main.py
│   └── apple_healthy5.jpg
└── README.md (this file)
```

## Installation

1. Clone the repository:
   
   ```bash
   git clone https://github.com/Bhumi2004/Apple-Tree-Health-Detection
   ```

2. Navigate to the project directory:

   ```bash
   cd Apple-Tree-Health-Detection
   ```

## Usage

### Apple Ripeness Detection

1. Navigate to the `Apple-Detection` folder:

   ```bash
   cd Apple Detection
   ```

2. Run the detection script:

   ```bash
   python apple.py
   ```

This module will analyze images of apples, categorize them as *ripe*, *unripe*, or *overripe*, and provide a total count of each type.

### Apple Leaf Detection

1. Navigate to the `apple-leaf-detection` folder:

   ```bash
   cd Apple Leaf Detection
   ```

2. Run the detection script:

   ```bash
   python app.py
   ```

This module will assess the health of apple leaves and determine whether they are *healthy* or affected by *scab disease*. Additionally, you 
can replace the image path in the script to test a specific apple leaf image of your choice. Please ensure that the provided image is of an apple leaf.

## Demo

Check out a video demonstration of the project: [Video Demo Link](https://youtu.be/qPibGYCpPHg?si=1nKQ9_Yl5aWV7JxK)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a pull request.

## License

This project is licensed under the MIT License.
```

