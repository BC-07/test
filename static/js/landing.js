class LandingPage {
  constructor() {
    this.init()
  }

  init() {
    this.setupThemeToggle()
    this.setupSmoothScrolling()
    this.hideLoadingScreen()
    this.setupAnimations()
  }

  hideLoadingScreen() {
    setTimeout(() => {
      const loadingScreen = document.getElementById("loadingScreen")
      if (loadingScreen) {
        loadingScreen.classList.add("hidden")
        setTimeout(() => {
          loadingScreen.style.display = "none"
        }, 500)
      }
    }, 1500)
  }

  setupThemeToggle() {
    const themeToggle = document.getElementById("themeToggle")
    const html = document.documentElement

    // Load saved theme
    const savedTheme = localStorage.getItem("theme") || "light"
    html.setAttribute("data-theme", savedTheme)
    this.updateThemeIcon(savedTheme)

    if (themeToggle) {
      themeToggle.addEventListener("click", () => {
        const currentTheme = html.getAttribute("data-theme")
        const newTheme = currentTheme === "light" ? "dark" : "light"

        html.setAttribute("data-theme", newTheme)
        localStorage.setItem("theme", newTheme)
        this.updateThemeIcon(newTheme)
      })
    }
  }

  updateThemeIcon(theme) {
    const themeToggle = document.getElementById("themeToggle")
    if (themeToggle) {
      const icon = themeToggle.querySelector("i")
      icon.className = theme === "light" ? "fas fa-moon" : "fas fa-sun"
    }
  }

  setupSmoothScrolling() {
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
  }

  setupAnimations() {
    // Intersection Observer for animations
    const observerOptions = {
      threshold: 0.1,
      rootMargin: "0px 0px -50px 0px",
    }

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("animate-fade-in-up")
        }
      })
    }, observerOptions)

    // Observe feature cards
    document.querySelectorAll(".feature-card").forEach((card) => {
      observer.observe(card)
    })

    // Observe about features
    document.querySelectorAll(".about-feature").forEach((feature) => {
      observer.observe(feature)
    })

    // Observe team members
    document.querySelectorAll(".team-member").forEach((member) => {
      observer.observe(member)
    })
  }
}

// Initialize the landing page
document.addEventListener("DOMContentLoaded", () => {
  new LandingPage()
})
