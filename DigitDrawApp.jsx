import React, { useRef, useState } from 'react';

const CANVAS_SIZE = 280; // Display size
const PIXEL_SIZE = 10; // Each MNIST pixel is 10x10
const MNIST_SIZE = 28; // 28x28 for MNIST

function pixelizeImage(ctx, mnistData) {
  // Draw the 28x28 data as 10x10 blocks
  for (let y = 0; y < MNIST_SIZE; y++) {
    for (let x = 0; x < MNIST_SIZE; x++) {
      const v = mnistData[y * MNIST_SIZE + x];
      ctx.fillStyle = v > 0.5 ? '#fff' : '#000';
      ctx.fillRect(x * PIXEL_SIZE, y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
    }
  }
}

export default function DigitDrawApp() {
  const [prediction, setPrediction] = useState(null);
  const [gptValidation, setGptValidation] = useState(null);
  const [drawing, setDrawing] = useState(false);
  const [mnistData, setMnistData] = useState(Array(MNIST_SIZE * MNIST_SIZE).fill(0));
  const canvasRef = useRef();

  // Draw on a hidden 28x28 canvas, then update the display
  const handlePointerDown = e => {
    setDrawing(true);
    drawAtEvent(e);
  };
  const handlePointerUp = () => setDrawing(false);
  const handlePointerMove = e => {
    if (drawing) drawAtEvent(e);
  };
  function drawAtEvent(e) {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = Math.floor(((e.touches ? e.touches[0].clientX : e.clientX) - rect.left) / PIXEL_SIZE);
    const y = Math.floor(((e.touches ? e.touches[0].clientY : e.clientY) - rect.top) / PIXEL_SIZE);
    if (x >= 0 && x < MNIST_SIZE && y >= 0 && y < MNIST_SIZE) {
      const newData = mnistData.slice();
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < MNIST_SIZE && ny >= 0 && ny < MNIST_SIZE) {
            newData[ny * MNIST_SIZE + nx] = 1;
          }
        }
      }
      setMnistData(newData);
      redraw(newData);
    }
  }
  function redraw(data) {
    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    pixelizeImage(ctx, data);
  }
  function clearCanvas() {
    setMnistData(Array(MNIST_SIZE * MNIST_SIZE).fill(0));
    redraw(Array(MNIST_SIZE * MNIST_SIZE).fill(0));
    setPrediction(null);
    setGptValidation(null);
  }
  async function predict() {
    // Convert mnistData to PNG
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = MNIST_SIZE;
    tempCanvas.height = MNIST_SIZE;
    const tctx = tempCanvas.getContext('2d');
    const imgData = tctx.createImageData(MNIST_SIZE, MNIST_SIZE);
    for (let i = 0; i < mnistData.length; i++) {
      const v = mnistData[i] > 0.5 ? 255 : 0;
      imgData.data[i * 4 + 0] = v;
      imgData.data[i * 4 + 1] = v;
      imgData.data[i * 4 + 2] = v;
      imgData.data[i * 4 + 3] = 255;
    }
    tctx.putImageData(imgData, 0, 0);
    const dataURL = tempCanvas.toDataURL('image/png');
    setPrediction('...');
    setGptValidation('...');
    try {
      const res = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataURL })
      });
      const data = await res.json();
      setPrediction(data.prediction !== undefined ? data.prediction : 'Error');
      setGptValidation(data.gpt_validation || '');
    } catch (e) {
      setPrediction('Error');
      setGptValidation('');
    }
  }
  // Redraw on mount and when mnistData changes
  React.useEffect(() => { redraw(mnistData); }, []);

  return (
    <div style={{
      minHeight: '100vh', background: '#1976d2', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center'
    }}>
      <h2 style={{ color: '#fff', fontWeight: 'bold', marginBottom: 24, textShadow: '2px 2px 8px #000' }}>Draw a digit (0-9)</h2>
      <div style={{
        background: '#fff', borderRadius: 32, boxShadow: '0 4px 32px #0004', padding: 24, display: 'flex', flexDirection: 'column', alignItems: 'center'
      }}>
        <canvas
          ref={canvasRef}
          width={CANVAS_SIZE}
          height={CANVAS_SIZE}
          style={{
            borderRadius: 24, border: '2px solid #1976d2', background: '#000', imageRendering: 'pixelated', cursor: 'crosshair', marginBottom: 16
          }}
          onMouseDown={handlePointerDown}
          onMouseUp={handlePointerUp}
          onMouseMove={handlePointerMove}
          onMouseLeave={handlePointerUp}
          onTouchStart={handlePointerDown}
          onTouchEnd={handlePointerUp}
          onTouchMove={handlePointerMove}
        />
        <div style={{ display: 'flex', gap: 16, marginTop: 8 }}>
          <button
            onClick={predict}
            style={{
              background: '#e53935', color: '#fff', border: 'none', borderRadius: 20, padding: '12px 32px', fontWeight: 'bold', fontSize: 18, boxShadow: '0 2px 8px #0003', cursor: 'pointer', transition: 'background 0.2s'
            }}
          >Predict</button>
          <button
            onClick={clearCanvas}
            style={{
              background: '#e53935', color: '#fff', border: 'none', borderRadius: 20, padding: '12px 32px', fontWeight: 'bold', fontSize: 18, boxShadow: '0 2px 8px #0003', cursor: 'pointer', transition: 'background 0.2s'
            }}
          >Clear</button>
        </div>
        <div style={{ color: '#1976d2', fontWeight: 'bold', fontSize: 28, marginTop: 24, textShadow: '1px 1px 8px #fff' }}>
          {prediction !== null && `Prediction: ${prediction}`}
        </div>
        <div style={{ color: '#e53935', fontWeight: 'bold', fontSize: 22, marginTop: 8, textShadow: '1px 1px 8px #fff' }}>
          {gptValidation && `GPT Validation: ${gptValidation}`}
        </div>
      </div>
    </div>
  );
} 