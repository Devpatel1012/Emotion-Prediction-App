// Emotion configuration with exact colors matching the UI
const emotionConfig = {
  sadness: {
    icon: "fas fa-frown",
    color: "#60a5fa",
    bgColor: "rgba(96, 165, 250, 0.2)",
  },
  joy: {
    icon: "fas fa-smile",
    color: "#fbbf24",
    bgColor: "rgba(251, 191, 36, 0.2)",
  },
  love: {
    icon: "fas fa-heart",
    color: "#f472b6",
    bgColor: "rgba(244, 114, 182, 0.2)",
  },
  anger: {
    icon: "fas fa-angry",
    color: "#ef4444",
    bgColor: "rgba(239, 68, 68, 0.2)",
  },
  fear: {
    icon: "fas fa-exclamation-triangle",
    color: "#a855f7",
    bgColor: "rgba(168, 85, 247, 0.2)",
  },
  surprise: {
    icon: "fas fa-surprise",
    color: "#10b981",
    bgColor: "rgba(16, 185, 129, 0.2)",
  },
}

// Modal functions
function openModal(modalType) {
  const modal = document.getElementById(modalType + "Modal")
  modal.classList.add("show")
  document.body.style.overflow = "hidden"
}

function closeModal(modalType) {
  const modal = document.getElementById(modalType + "Modal")
  modal.classList.remove("show")
  document.body.style.overflow = "auto"
}

// Scroll to section function
function scrollToSection(sectionId) {
  const section = document.getElementById(sectionId)
  if (section) {
    section.scrollIntoView({ behavior: "smooth" })
  }
}

// Close modal when clicking outside
document.addEventListener("click", (event) => {
  if (event.target.classList.contains("modal-overlay")) {
    const modalId = event.target.id
    const modalType = modalId.replace("Modal", "")
    closeModal(modalType)
  }
})

// Close modal with Escape key
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    const openModal = document.querySelector(".modal-overlay.show")
    if (openModal) {
      const modalType = openModal.id.replace("Modal", "")
      closeModal(modalType)
    }
  }
})

// Main emotion classification function - connects to your Flask backend
async function classifyEmotion() {
  const inputText = document.getElementById("inputText").value
  const analyzeBtn = document.getElementById("analyzeBtn")
  const btnText = document.getElementById("btnText")
  const errorMessage = document.getElementById("errorMessage")
  const resultsCard = document.getElementById("resultsCard")

  // Clear previous results
  hideError()
  hideResults()

  if (!inputText.trim()) {
    showError("Please enter some text to analyze.")
    return
  }

  // Show loading state
  setLoadingState(true)

  try {
    // Call your Flask backend /predict endpoint
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: inputText }),
    })

    const result = await response.json()

    if (response.ok) {
      displayResults(result)
    } else {
      showError(result.error || "An unknown error occurred.")
    }
  } catch (error) {
    console.error("Error:", error)
    showError("Could not connect to the server. Please try again.")
  } finally {
    setLoadingState(false)
  }
}

// Helper functions
function setLoadingState(loading) {
  const analyzeBtn = document.getElementById("analyzeBtn")
  const btnText = document.getElementById("btnText")

  analyzeBtn.disabled = loading

  if (loading) {
    btnText.innerHTML = '<div class="loading-spinner"></div> Analyzing...'
  } else {
    btnText.innerHTML = '<i class="fas fa-bolt"></i> Classify Emotion'
  }
}

function showError(message) {
  const errorMessage = document.getElementById("errorMessage")
  errorMessage.textContent = message
  errorMessage.classList.add("show")
}

function hideError() {
  const errorMessage = document.getElementById("errorMessage")
  errorMessage.classList.remove("show")
}

function hideResults() {
  const resultsCard = document.getElementById("resultsCard")
  resultsCard.classList.remove("show")
}

function displayResults(result) {
  const resultsCard = document.getElementById("resultsCard")
  const resultIcon = document.getElementById("resultIcon")
  const resultEmotion = document.getElementById("resultEmotion")
  const resultConfidence = document.getElementById("resultConfidence")
  const probabilitiesList = document.getElementById("probabilitiesList")

  // Update main result
  const emotion = result.predicted_emotion
  const confidence = (Number.parseFloat(result.confidence) * 100).toFixed(1)
  const config = emotionConfig[emotion]

  // Update main emotion display
  resultIcon.className = config.icon
  resultIcon.style.color = config.color
  resultEmotion.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1)
  resultEmotion.style.color = config.color
  resultConfidence.textContent = confidence + "%"

  // Update the icon container background
  const iconContainer = resultIcon.parentElement
  iconContainer.style.background = `linear-gradient(135deg, ${config.bgColor}, rgba(255, 255, 255, 0.1))`

  // Update probabilities with styling to match the exact image design
  probabilitiesList.innerHTML = ""

  // Sort emotions by probability (descending) to match your image
  const sortedEmotions = Object.entries(result.all_probabilities).sort(([, a], [, b]) => b - a)

  sortedEmotions.forEach(([emotionName, probability]) => {
    const percentage = (probability * 100).toFixed(1)
    const config = emotionConfig[emotionName]

    const probabilityItem = document.createElement("div")
    probabilityItem.className = "probability-item"

    probabilityItem.innerHTML = `
      <div class="probability-label">
        <i class="${config.icon}" style="color: ${config.color}"></i>
        <span>${emotionName.charAt(0).toUpperCase() + emotionName.slice(1)}</span>
      </div>
      <div class="probability-bar">
        <div class="probability-fill" style="width: 0%; background: ${config.color}; transition: width 1.2s ease-out;"></div>
      </div>
      <div class="probability-value">${percentage}%</div>
    `

    probabilitiesList.appendChild(probabilityItem)

    // Animate the bar after a short delay to create the filling effect
    setTimeout(() => {
      const fill = probabilityItem.querySelector(".probability-fill")
      fill.style.width = percentage + "%"
    }, 200)
  })

  resultsCard.classList.add("show")
}

// Event listeners
document.addEventListener("DOMContentLoaded", () => {
  // Add keyboard shortcut for analysis (Ctrl+Enter)
  document.getElementById("inputText").addEventListener("keydown", (event) => {
    if (event.ctrlKey && event.key === "Enter") {
      classifyEmotion()
    }
  })

  // Add smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault()
      const target = document.querySelector(this.getAttribute("href"))
      if (target) {
        target.scrollIntoView({
          behavior: "smooth",
          block: "start",
        })
      }
    })
  })

  // Add some example texts for demonstration
  const examples = [
    "I'm so excited about my vacation tomorrow!",
    "I can't believe you would do this to me.",
    "I miss you so much, my heart aches.",
    "This is the best day of my life!",
    "I'm really worried about the exam results.",
    "Wow, I never expected this to happen!",
  ]

  // Optional: Add example text cycling functionality
  const exampleIndex = 0
  const textInput = document.getElementById("inputText")

  // You can uncomment this to add example cycling on focus
  /*
  textInput.addEventListener("focus", () => {
    if (!textInput.value.trim()) {
      textInput.placeholder = examples[exampleIndex % examples.length];
      exampleIndex++;
    }
  });
  */
})

// Add intersection observer for animations
const observerOptions = {
  threshold: 0.1,
  rootMargin: "0px 0px -50px 0px",
}

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = "1"
      entry.target.style.transform = "translateY(0)"
    }
  })
}, observerOptions)

// Observe elements for scroll animations
document.addEventListener("DOMContentLoaded", () => {
  const animatedElements = document.querySelectorAll(".feature-card, .tech-specs, .research-card")
  animatedElements.forEach((el) => {
    el.style.opacity = "0"
    el.style.transform = "translateY(20px)"
    el.style.transition = "opacity 0.6s ease, transform 0.6s ease"
    observer.observe(el)
  })
})
