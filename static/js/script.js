// loading animation for prediction

document.querySelectorAll("form").forEach(form => {

form.addEventListener("submit", () => {

let btn = form.querySelector("button");

btn.innerHTML = "Analyzing MRI...";
btn.disabled = true;

});

});