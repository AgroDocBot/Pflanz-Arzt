document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('submit-button').addEventListener('click', function() {
        const fileInput = document.getElementById('file-input');
        const file = fileInput.files[0];

        if (!file) {
            alert('Please upload an image file.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        const reader = new FileReader();
        reader.onload = function(event) {
            const imgElement = document.getElementById('uploaded-image');
            imgElement.src = event.target.result;
            console.log(event.target.result);
            imgElement.style.display = 'block';
        };
        reader.readAsDataURL(file);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            const predictionDiv = document.getElementById('prediction');
            const falseDiv = document.getElementById('false_btn');
            if (data.predicted_class) {
                predictionDiv.innerHTML = `Predicted Class: <strong>${data.predicted_class}</strong>`;
                falseDiv.innerHTML = `Class not correct?`;
            } else {
                predictionDiv.innerHTML = `Error: ${data.error}`;
            }
            predictionDiv.style.display = 'block';
            falseDiv.style.display = 'block';
        })
        .catch(error => {
            console.error('Error:', error);
            alert('There was an error processing your request.');
        });

    });


    document.getElementById('update').addEventListener('click', () => {
        fetch('/update-model', {
            method : 'GET'
        })
    })

    document.getElementById('false_btn').addEventListener('click', function() {
        const fileInput = document.getElementById('file-input');
        const file = fileInput.files[0];

        if (!file) {
            alert('Please upload an image file.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        const reader = new FileReader();
        reader.onload = function(event) {
            const imgElement = document.getElementById('uploaded-image');
            imgElement.src = event.target.result;
            console.log(event.target.result);
            imgElement.style.display = 'block';
        };
        reader.readAsDataURL(file);

        fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.text(); 
            })
            .then(html => {
                document.open(); 
                document.write(html); 
                document.close(); 
                //window.location.href = '/result';
            })
            .catch(error => {
                console.error('Error:', error);
                alert('There was an error processing your request.');
            })

    })
})