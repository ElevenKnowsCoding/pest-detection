# Agricultural Pest Counter

A web-based application for detecting and counting pests and eggs on leaves using computer vision techniques.

## Features

- **Authentication**: Secure login with authentication key
- **Image Upload**: Upload existing images of leaves with pests
- **Camera Scanning**: Use your device's camera to capture and analyze images in real-time
- **Pest & Egg Detection**: Automatically detect and count both active pests and eggs
- **Visual Results**: See detected pests (red circles) and eggs (orange circles) with numbered markers
- **Leaf Structure Analysis**: Proper leaf outline and vein detection
- **Shadow Removal**: Excludes border shadows for accurate counting

## Detection Details

- **Red Circles**: Active pests (size ≥ 150px)
- **Orange Circles**: Eggs (size < 150px)
- **Green Outline**: Leaf boundary
- **Border Exclusion**: 25px erosion to avoid shadow detection

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set your authentication keys:
   - `AUTH_KEY`: Your login password
   - `SECRET_KEY`: Flask session secret

4. **Run the Application**:
   ```bash
   python app.py
   ```

5. **Access the Application**:
   - Open your web browser and go to `http://localhost:5002`
   - Login with your `AUTH_KEY`

## Usage

### Upload Mode
1. Click on "Upload Analysis" tab
2. Drag and drop an image or click "Select Images"
3. Click "Analyze for Pests" to process
4. View results showing pest count, egg count, and detected locations

### Camera Mode
1. Click on "Live Scanning" tab
2. Click "Start Camera" to activate your camera
3. Position the camera over the leaf
4. Click "Capture & Analyze" to take a photo and process it
5. View results with pest and egg counts

## Technical Details

The application uses:
- **Flask** for the web framework
- **OpenCV** for image processing and computer vision
- **NumPy** for numerical operations
- **PIL** for image handling
- **python-dotenv** for environment variable management

### Detection Algorithm
The pest detection uses:
1. HSV color space conversion to identify dark spots
2. Leaf mask extraction with green color detection
3. Vein structure extraction using morphological operations
4. Border erosion to exclude shadow areas
5. Contour detection to find potential pests
6. Size-based classification (eggs vs active pests)
7. Morphological operations to clean noise

## Deployment

### Vercel Deployment

See `DEPLOYMENT.md` for detailed Vercel deployment instructions.

### Environment Variables

For production deployment, set these environment variables:
- `AUTH_KEY`: Your authentication key
- `SECRET_KEY`: Flask secret key for sessions

## File Structure

```
├── app.py              # Main Flask application
├── auth.py             # Authentication middleware
├── detect.py           # Pest/egg detection logic
├── leaf_structure.py   # Leaf outline and vein extraction
├── templates/
│   ├── index.html      # Main web interface
│   └── login.html      # Login page
├── uploads/            # Uploaded and processed images
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore rules
├── vercel.json         # Vercel deployment config
├── DEPLOYMENT.md       # Deployment instructions
└── README.md           # This file
```

## Security

- Change the default `AUTH_KEY` in `.env` file
- Never commit `.env` file to GitHub
- Use strong secret keys in production
- The `.env` file is excluded via `.gitignore`

## Customization

To improve detection accuracy:
1. Adjust color ranges in `detect.py`
2. Modify size thresholds (currently 150px for egg/pest boundary)
3. Fine-tune border erosion (currently 25px)
4. Adjust minimum detection size (currently 55px)

## Future Enhancements

- Machine learning model integration
- Pest species classification
- Batch processing for multiple images
- Export results to CSV/PDF
- Mobile app version
- Database storage for analysis history
- Real-time video analysis

## License

MIT License