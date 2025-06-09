// --- Hilfsfunktionen zur Daten-Erzeugung ---
function trueFunction(x) {
  return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

function generateData(n = 100, noiseVar = 0.05) {
  const x = Array.from({ length: n }, () => Math.random() * 4 - 2);
  const yTrue = x.map(trueFunction);
  const yNoisy = yTrue.map(y => y + randn(0, Math.sqrt(noiseVar)));

  const combined = x.map((xi, i) => ({ x: xi, y: yTrue[i], yNoisy: yNoisy[i] }));
  shuffleArray(combined);

  const train = combined.slice(0, n / 2);
  const test = combined.slice(n / 2);

  return {
    trainClean: train.map(p => [p.x, p.y]),
    testClean: test.map(p => [p.x, p.y]),
    trainNoisy: train.map(p => [p.x, p.yNoisy]),
    testNoisy: test.map(p => [p.x, p.yNoisy]),
  };
}

function randn(mu = 0, sigma = 1) {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return sigma * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) + mu;
}

function shuffleArray(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

// --- TensorFlow Modell und MSE ---
async function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [1] }));
  model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1 })); // Linear output
  return model;
}

async function trainModel(model, xTrain, yTrain, epochs = 100) {
  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'meanSquaredError'
  });

  await model.fit(xTrain, yTrain, {
    epochs: epochs,
    batchSize: 32,
    shuffle: true,
    verbose: 0
  });

  return model;
}

function predict(model, x) {
  const xTensor = tf.tensor2d(x, [x.length, 1]);
  const yPred = model.predict(xTensor);
  return yPred.dataSync();
}

function toTensor(data) {
  const x = data.map(d => d[0]);
  const y = data.map(d => d[1]);
  return {
    x: tf.tensor2d(x, [x.length, 1]),
    y: tf.tensor2d(y, [y.length, 1])
  };
}

function calculateMSE(yTrue, yPred) {
  const errors = yTrue.map((y, i) => (y - yPred[i]) ** 2);
  return errors.reduce((a, b) => a + b, 0) / yTrue.length;
}

// --- Plotly Visualisierung ---
function plotData(divId, trainData, testData, title) {
  const trainTrace = {
    x: trainData.map(p => p[0]),
    y: trainData.map(p => p[1]),
    mode: 'markers',
    name: 'Train',
    marker: { color: 'blue' }
  };

  const testTrace = {
    x: testData.map(p => p[0]),
    y: testData.map(p => p[1]),
    mode: 'markers',
    name: 'Test',
    marker: { color: 'red' }
  };

  Plotly.newPlot(divId, [trainTrace, testTrace], {
    title: title,
    margin: { t: 40 }
  });
}

function plotPrediction(divId, x, yTrue, yPred, labelTrue = 'True', labelPred = 'Predicted') {
  const trueTrace = {
    x: x,
    y: yTrue,
    mode: 'markers',
    name: labelTrue,
    marker: { color: 'gray' }
  };

  const predTrace = {
    x: x,
    y: yPred,
    mode: 'lines',
    name: labelPred,
    line: { color: 'green' }
  };

  Plotly.newPlot(divId, [trueTrace, predTrace], {
    title: `${labelPred} vs ${labelTrue}`,
    margin: { t: 40 }
  });
}

// --- Hauptfunktion ---
async function run() {
  const data = generateData(100, 0.05);

  // Plot R1
  plotData("r1-left", data.trainClean, data.testClean, "Unverrauschte Daten");
  plotData("r1-right", data.trainNoisy, data.testNoisy, "Verrauschte Daten");

  // Modell 1: Ohne Rauschen trainiert
  const modelClean = await createModel();
  const tensors1 = toTensor(data.trainClean);
  await trainModel(modelClean, tensors1.x, tensors1.y, 200);

  const xTrainClean = data.trainClean.map(p => p[0]);
  const xTestClean = data.testClean.map(p => p[0]);
  const yTrainClean = data.trainClean.map(p => p[1]);
  const yTestClean = data.testClean.map(p => p[1]);

  const predTrain1 = predict(modelClean, xTrainClean);
  const predTest1 = predict(modelClean, xTestClean);

  plotPrediction("r2-left", xTrainClean, yTrainClean, predTrain1);
  plotPrediction("r2-right", xTestClean, yTestClean, predTest1);

  const lossTrain1 = calculateMSE(yTrainClean, predTrain1).toFixed(4);
  const lossTest1 = calculateMSE(yTestClean, predTest1).toFixed(4);
  document.getElementById("loss-r2").innerText = `MSE Train: ${lossTrain1}, MSE Test: ${lossTest1}`;

  // Modell 2: Best-Fit mit Rauschen (z. B. 100 Epochen)
  const modelBest = await createModel();
  const tensors2 = toTensor(data.trainNoisy);
  await trainModel(modelBest, tensors2.x, tensors2.y, 100);

  const xTrainNoisy = data.trainNoisy.map(p => p[0]);
  const xTestNoisy = data.testNoisy.map(p => p[0]);
  const yTrainNoisy = data.trainNoisy.map(p => p[1]);
  const yTestNoisy = data.testNoisy.map(p => p[1]);

  const predTrain2 = predict(modelBest, xTrainNoisy);
  const predTest2 = predict(modelBest, xTestNoisy);

  plotPrediction("r3-left", xTrainNoisy, yTrainNoisy, predTrain2);
  plotPrediction("r3-right", xTestNoisy, yTestNoisy, predTest2);

  const lossTrain2 = calculateMSE(yTrainNoisy, predTrain2).toFixed(4);
  const lossTest2 = calculateMSE(yTestNoisy, predTest2).toFixed(4);
  document.getElementById("loss-r3").innerText = `MSE Train: ${lossTrain2}, MSE Test: ${lossTest2}`;

  // Modell 3: Overfit mit Rauschen (z. B. 1000 Epochen)
  const modelOverfit = await createModel();
  await trainModel(modelOverfit, tensors2.x, tensors2.y, 1000);

  const predTrain3 = predict(modelOverfit, xTrainNoisy);
  const predTest3 = predict(modelOverfit, xTestNoisy);

  plotPrediction("r4-left", xTrainNoisy, yTrainNoisy, predTrain3);
  plotPrediction("r4-right", xTestNoisy, yTestNoisy, predTest3);

  const lossTrain3 = calculateMSE(yTrainNoisy, predTrain3).toFixed(4);
  const lossTest3 = calculateMSE(yTestNoisy, predTest3).toFixed(4);
  document.getElementById("loss-r4").innerText = `MSE Train: ${lossTrain3}, MSE Test: ${lossTest3}`;

  // Diskussion automatisch einfügen
  document.getElementById("discussion").innerText =
    `Ohne Rauschen ist der MSE auf Train und Test nahezu identisch. ` +
    `Das Best-Fit Modell mit Rauschen hat einen guten Kompromiss zwischen Bias und Varianz. ` +
    `Das Overfitted Modell zeigt deutlich geringeren Train-Loss, aber schlechtere Generalisierung.`;
}

run();
