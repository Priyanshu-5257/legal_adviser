# Legal Agreement Clauses Bot

## Overview
The Legal Agreement Clauses Bot is a Python application built with Flask and Google Palm Language Model to assist users in extracting important missing clauses from legal agreements. The bot takes legal agreements in DOCX format as input and returns a DOCX file that contains the missing important clauses based on the given agreement.

## Installation
1. Clone the repository:
   ```
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up the Google Palm API key:
   - Create a `.env` file in the project root directory.
   - Add your Google Palm API key to the `.env` file:
     ```
     API_KEY=<your_google_palm_api_key>
     ```

4. Run the application:
   ```
   python app.py
   ```

## Usage
1. Access the bot's web interface by navigating to `http://localhost:5000` in your web browser.

2. Upload a legal agreement in DOCX format using the provided file input field.

3. Click the "Submit" button to process the document and generate the output file.

4. The bot will extract important missing clauses from the uploaded agreement and create a new DOCX file containing the generated clauses.

## Guidelines
- The bot assumes that the input legal agreement is in DOCX format.
- Ensure your Google Palm API key is valid and available in the `.env` file.
- The output DOCX file will be named "output.docx" and will be available for download.

## Note
Please keep in mind that this bot is a demonstration of using the Google Palm Language Model and may not be suitable for production use. The extracted clauses are generated based on the model's predictions and may not always be accurate or legally sound. Always consult with legal experts for reliable advice and agreement analysis.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
