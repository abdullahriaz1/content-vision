const express = require('express');
const fs = require('fs');
const path = require('path');
const cors = require('cors');
const multer = require('multer');

const app = express();
const port = 3000;

app.use(cors());

const videosDir = path.join(__dirname, 'videos');

// Set up multer for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, videosDir);
    },
    filename: (req, file, cb) => {
        cb(null, file.originalname);
    }
});

const upload = multer({ storage: storage });

app.get('/video/videoList', (req, res) => {
  fs.readdir(videosDir, (err, files) => {
      if (err) {
          return res.status(500).send('Unable to scan directory: ' + err);
      }
      const videoList = files.filter(file => path.extname(file) === '.mp4').map(file => encodeURIComponent(file));
      res.json(videoList);
  });
});

app.get('/video/:videoName', (req, res) => {
  // Decode the URI component of videoName
  const videoName = req.params.videoName;
  const videoPath = path.join(videosDir, videoName);

  if (!fs.existsSync(videoPath)) {
      return res.status(404).send('Video not found');
  }

  const videoStat = fs.statSync(videoPath);
  const fileSize = videoStat.size;
  const range = req.headers.range;

  if (range) {
      const parts = range.replace(/bytes=/, "").split("-");
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;

      if (start >= fileSize) {
          res.status(416).send('Requested range not satisfiable\n' + start + ' >= ' + fileSize);
          return;
      }

      const chunksize = (end - start) + 1;
      const file = fs.createReadStream(videoPath, { start, end });
      const head = {
          'Content-Range': `bytes ${start}-${end}/${fileSize}`,
          'Accept-Ranges': 'bytes',
          'Content-Length': chunksize,
          'Content-Type': 'video/mp4',
      };

      res.writeHead(206, head);
      file.pipe(res);
  } else {
      const head = {
          'Content-Length': fileSize,
          'Content-Type': 'video/mp4',
      };
      res.writeHead(200, head);
      fs.createReadStream(videoPath).pipe(res);
  }
});

app.put('/video/upload', upload.single('video'), (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).send('No file uploaded.');
        }

        // Decode the filename if necessary
        const originalFilename = decodeURIComponent(req.file.originalname);
        const destinationPath = path.join('videos', originalFilename);

        fs.rename(req.file.path, destinationPath, (err) => {
            if (err) {
                return res.status(500).send('Error saving file.');
            }
            res.status(200).send('File uploaded successfully.');
        });
    } catch (error) {
        console.error('Error processing upload:', error);
        res.status(500).send('Server error');
    }
});



app.listen(port, () => {
    console.log(`Video server running at http://localhost:${port}`);
});
