# Smart Translation Application

## Overview

This application is a smart translation tool, similar in structure to other translation app. It leverages AI models to identify and translate entities within text, enhancing translation accuracy and context understanding.

## Installation Instructions

1. **Set Server Adress**
   ```
   Create a `.env` file in the root directory and add necessary environment variables as specified in the `.env.example`.
   ```

There are two options to run the app, run locally or using docker.

2. **Using Docker**
   - Ensure Docker is installed and running on your system.
   - Build and run the Docker containers:
     ```bash
     docker-compose up --build
     ```

3. **Using Locally**

3.1. **Install dependencies**:
   ```bash
   npm install
   ```

3.2. **Set up environment variables**:
   Create a `.env` file in the root directory and add necessary environment variables as specified in the `.env.example` (if available).

3.3. **Run the application**:
   ```bash
   npm start
   ```

4. **Run the App**:
   The application should now be running and accessible at `http://localhost:3000`.

## Folder Structure

Here's an overview of the project's folder structure:

- **node_modules/**: Contains all the npm packages installed for the project.
- **public/**: Public assets like the `index.html` file.
- **src/**: Contains the source code for the React application.

  - **Commons/**: Holds reusable components utilized throughout the application.
  - **Components/**: Contains all the main components of the application.
    - **Header/**: Components related to the header section of the app.
    - **Main/**: Main application components.
  - **service/**: Manages server requests and helper functions.
    - **request.ts**: Handles API requests to the server.
    - **helper.ts**: Contains utility functions used across the application.
  - **Types/**: Houses TypeScript interfaces and types.
  - **App.css**: Main CSS file for the app's styling.
  - **App.tsx**: Entry point for the React application.
  - **index.css**: Global CSS styles.
  - **index.tsx**: Main entry file rendering the React app.
  - **react-app-env.d.ts**: TypeScript declaration file.
  - **reportWebVitals.ts**: Measures and reports web vitals.
  - **setupTests.ts**: Configures the testing environment.

- **.env**: Environment variables for the project.
- **.gitignore**: Specifies files and directories to be ignored by Git.
- **docker-compose.yml**: Configuration file for Docker Compose.
- **Dockerfile**: Contains instructions for building a Docker image.
- **package-lock.json**: Automatically generated file holding the exact versions of npm dependencies.
- **package.json**: Lists the project's dependencies and scripts.
- **README.md**: This file.
- **tsconfig.json**: TypeScript configuration file.
