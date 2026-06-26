async function predictScore() {
    let hours = document.getElementById("hours").value;
    let loader = document.getElementById("loader");

    if (!hours) {
        document.getElementById("result").innerText = "Please enter hours";
        return;
    }

    loader.style.display = "inline-block";

    let response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ hours: hours })
    });

    let data = await response.json();

    loader.style.display = "none";

    document.getElementById("result").innerText =
        "Predicted Score: " + data.predicted_score;
}