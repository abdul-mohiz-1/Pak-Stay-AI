// Step 2 -> Step 3 (Search & Load)
    searchBtn.addEventListener('click', async () => {
        switchStep(step2, stepLoading);
        
        // Frontend se values uthana
        const payload = {
            city: currentCity,
            travel_type: document.getElementById('travel-type').value,
            budget: document.getElementById('budget-slider').value
        };

        try {
            // Flask Backend ko data bhejna
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const aiHotelsData = await response.json(); // Backend se JSON lena
            
            // Screen update karna
            switchStep(stepLoading, stepResults);
            renderCards(aiHotelsData); // Naya data render function ko dena
            
        } catch (error) {
            console.error("Error fetching data:", error);
            alert("Something went wrong! Backend se connection nahi hua.");
            switchStep(stepLoading, step2);
        }
    });

    // Generate Cards (Ab yeh fake data nahi, backend se data lega)
    function renderCards(hotelDataArray) {
        const container = document.getElementById('hotel-cards-container');
        container.innerHTML = ''; 

        hotelDataArray.forEach((hotel, index) => {
            const animDelay = index * 0.2; 
            const badge = hotel.top ? `<span class="top-badge"><i class="fa-solid fa-crown"></i> Top Pick</span>` : '';
            const topClass = hotel.top ? 'top-choice' : '';
            
            // Handle amenities array safely
            let pills = '';
            if(Array.isArray(hotel.amenities)) {
                pills = hotel.amenities.map(a => `<span class="pill">${a}</span>`).join('');
            }

            const cardHTML = `
                <div class="hotel-card ${topClass}" style="animation-delay: ${animDelay}s">
                    ${badge}
                    <h3 class="hotel-name">${hotel.name}</h3>
                    <div class="hotel-meta">
                        <span><i class="fa-solid fa-location-dot"></i> ${currentCity}</span>
                        <span class="stars"><i class="fa-solid fa-star"></i> ${hotel.rating}/5</span>
                    </div>
                    <div class="price-badge">Rs. ${hotel.price} / night</div>
                    <div class="amenities">${pills}</div>
                    <div class="ai-comment"><i class="fa-solid fa-wand-magic-sparkles"></i> <strong>AI Insight:</strong> ${hotel.comment}</div>
                </div>
            `;
            container.innerHTML += cardHTML;
        });
    }