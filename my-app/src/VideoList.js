import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import FileUpload from './FileUpload';
const VideoList = () => {
    const [videoList, setVideoList] = useState([]);

    useEffect(() => {
      fetch('http://localhost:3000/video/videoList')
          .then(response => response.json())
          .then(data => setVideoList(data))
          .catch(error => console.error('Error fetching video list:', error));
  }, []);

  return (
      <div>
          <h1>Available Videos</h1>
          <ul>
              {videoList.map((video, index) => (
                  <li key={index}>
                      <Link to={`/video/${video}`}>
                          {decodeURIComponent(decodeURIComponent(video))}
                      </Link>
                  </li>
              ))}
          </ul>

            <FileUpload />
        </div>
    );
};

export default VideoList;
