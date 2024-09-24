import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import './VideoPlayer.css'; // Import custom styles

const VideoPlayer = () => {
    const { videoName } = useParams();
    const [videoUrl, setVideoUrl] = useState('');

    useEffect(() => {
        const encodedVideoName = encodeURIComponent(videoName);
        const url = `http://localhost:3000/video/${encodedVideoName}`;
        setVideoUrl(url);
    }, [videoName]);

    return (
        <div>
            <h1>Playing: 
                <br/>
                {(decodeURIComponent(videoName))}
            </h1>
            {videoUrl ? (
                <video controls width="100%">
                    <source src={videoUrl} type="video/mp4" />
                    Your browser does not support the video tag.
                </video>
            ) : (
                <p>Loading...</p>
            )}
        </div>
    );
};


export default VideoPlayer;
