import React, { useState } from 'react';
import axios from 'axios';

const FileUpload = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploadStatus, setUploadStatus] = useState('');

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    const handleFileUpload = async () => {
        if (!selectedFile) {
            setUploadStatus('Please select a file to upload.');
            return;
        }

        // Encode the filename
        const encodedFileName = encodeURIComponent(selectedFile.name);

        const formData = new FormData();
        formData.append('video', selectedFile, encodedFileName);

        try {
            setUploadStatus('Uploading...');
            await axios.put('http://localhost:3000/video/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setUploadStatus('File uploaded successfully.');
            setSelectedFile(null); // Clear the file input after upload
        } catch (error) {
            console.error('Error uploading file:', error);
            setUploadStatus('Failed to upload file.');
        }
    };

    return (
        <div>
            <h1>Upload a Video</h1>
            <input type="file" accept="video/mp4" onChange={handleFileChange} />
            <button onClick={handleFileUpload}>Upload</button>
            <p>{uploadStatus}</p>
        </div>
    );
};

export default FileUpload;
