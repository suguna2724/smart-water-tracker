async function getPrediction() {
  const country = document.getElementById("country").value;
  const year = document.getElementById("year").value;
  const predictionEl = document.getElementById("prediction");
  const insightEl = document.getElementById("insight");

  predictionEl.innerText = "Fetching prediction...";
  insightEl.innerText = "";

  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ country, year }),
  });

  const data = await response.json();
  predictionEl.innerHTML = `🌊 Predicted Water Consumption for <b>${country}</b> in <b>${year}</b>: 
                            ${data.prediction.toFixed(2)} Billion m³`;

  insightEl.innerText = data.insight;

  // Optional chart
  const ctx = document.getElementById("chart").getContext("2d");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Water Consumption"],
      datasets: [{
        label: `${country} (${year})`,
        data: [data.prediction],
        backgroundColor: "#0288d1",
      }]
    }
  });
}
