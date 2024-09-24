import React from 'react'
import video from './sample-5s.mp4'

function Main() {
  return (
    <div>
      
      Main
      <div className="video-player">
        <video width="1920" height="1080" controls >
          <source src={video} type="video/mp4"/>
        </video>
      </div>

    </div>

  )
}

export default Main