;(async () => {
  const model = await tf.loadGraphModel("/model/model.json");

//  const threshold = 0.75;
  const threshold = 0.5;
  const classesDir = {
    1: {
      name: 'Suzanne',
      id: 1,
    },
  }

  const mainCanvas = document.createElement("canvas");
  const ctx = mainCanvas.getContext("2d");
  const font = "16px sans-serif";
  ctx.font = font;
  ctx.textBaseline = "top";

  const inputElem = document.createElement("input")
  inputElem.type = "file"
  inputElem.accept = "image/*"
  inputElem.addEventListener("change", (e) => {
    const  files = e.target.files;
    const reader = new FileReader();
    reader.onload = (ee) => {
      const img = new Image()
      img.src = ee.target.result
      img.onload = () => {
        predict(img)
      }
    }
    reader.readAsDataURL(files[0]);
  })
  document.body.appendChild(inputElem)
  document.body.appendChild(mainCanvas)

  async function predict(img) {
    tf.engine().startScope()
    mainCanvas.width = img.width;
    mainCanvas.height = img.height;
    const tfImg = tf.browser.fromPixels(img).toInt();
    const expandedImg = tfImg.transpose([0, 1, 2]).expandDims();
    const predictions = await model.executeAsync(expandedImg);
    /*
    for(let i = 0; i < 8; i++) {
      console.log(predictions[i].arraySync());
      console.log(predictions[i])
    }
    */
    const boxes = predictions[2].arraySync(); // shape [0, 100, 4]
    const scores = predictions[5].arraySync(); // shape [1, 100]
    const classes = predictions[1].dataSync(); // shape [1, 100]
    const detectionObjects = []
    ctx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);
    ctx.drawImage(img, 0, 0, mainCanvas.width, mainCanvas.height);

    scores[0].forEach((score, i) => {
      if (score > threshold) {
        const bbox = [];
        const minY = boxes[0][i][0] * img.height;
        const minX = boxes[0][i][1] * img.width;
        const maxY = boxes[0][i][2] * img.height;
        const maxX = boxes[0][i][3] * img.width;
        bbox[0] = minX;
        bbox[1] = minY;
        bbox[2] = maxX - minX;
        bbox[3] = maxY - minY;
        const c = classesDir[classes[i]]
        detectionObjects.push({
          class: classes[i],
          label: c ? c.name : 'Unknown',
          score: score.toFixed(4),
          bbox: bbox
        })
      }
    })
    detectionObjects.forEach(item => {
      const x = item['bbox'][0];
      const y = item['bbox'][1];
      const width = item['bbox'][2];
      const height = item['bbox'][3];

      // Draw the bounding box.
      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, width, height);

      // Draw the label background.
      ctx.fillStyle = "#00FFFF";
      const scoreText = item["label"] + " " + (100*item["score"]).toFixed(2) + "%";
      const textWidth = ctx.measureText(scoreText).width;
      const textHeight = 16
      ctx.fillRect(x, y, textWidth, textHeight);
      ctx.fillStyle = "#000000";
      ctx.fillText(scoreText, x, y);
    });
    console.log(detectionObjects)
    tf.engine().endScope()
  }
})();