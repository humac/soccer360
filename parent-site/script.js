const app = {
    init() {
        this.bindEvents();
    },
    
    bindEvents() {
        // Nav links
        document.querySelectorAll('.nav-links a').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                document.querySelectorAll('.nav-links a').forEach(l => l.classList.remove('active'));
                e.target.classList.add('active');
                
                if(e.target.dataset.view === 'season') {
                    this.closeGame();
                } else {
                    alert('Documentation View (Not implemented in demo)');
                }
            });
        });

        // Player controls
        document.querySelectorAll('.player-controls button').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.player-controls button').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                
                // Update player label text mapping
                const label = document.querySelector('.player-label');
                const isTactical = e.target.textContent.toLowerCase().includes('tactical');
                label.textContent = isTactical ? 'TACTICAL_WIDE.MP4' : 'BROADCAST.MP4 (AUTO-FOLLOW)';
                
                // Visual feedback
                const indicator = document.querySelector('.play-indicator');
                indicator.style.transform = 'scale(0.9)';
                setTimeout(() => {
                    indicator.style.transform = 'scale(1)';
                }, 150);
            });
        });

        // Highlights
        document.querySelectorAll('.highlight-item').forEach(item => {
            item.addEventListener('click', () => {
                const activeBtn = document.querySelector('.player-controls button.active');
                if(!activeBtn.textContent.includes('Broadcast')) {
                    // switch to broadcast for highlights
                    document.querySelectorAll('.player-controls button')[0].click();
                }
                const label = document.querySelector('.player-label');
                const time = item.querySelector('.h-time').textContent;
                label.textContent = `SEEKING TO ${time}...`;
                setTimeout(() => {
                    label.textContent = `HIGHLIGHT.MP4 (PLAYING)`;
                }, 800);
            });
        });
    },

    openGame(id) {
        // Fetch game details based on id
        const details = {
            'game-01': { title: 'CITY VS UNITED', score: '3 - 1', date: 'Oct 14, 2026' },
            'game-02': { title: 'ROVERS VS CITY', score: '0 - 2', date: 'Oct 09, 2026' },
            'game-03': { title: 'CITY VS ATHLETIC', score: '1 - 1', date: 'Sep 30, 2026' },
            'game-04': { title: 'WANDERERS VS CITY', score: '0 - 4', date: 'Sep 22, 2026' }
        };

        if (details[id]) {
            document.getElementById('active-game-title').textContent = details[id].title;
            document.getElementById('active-game-score').textContent = details[id].score;
            document.getElementById('active-game-date').textContent = details[id].date;
        }

        document.getElementById('season-view').classList.remove('active');
        document.getElementById('game-view').classList.add('active');
        
        // ensure default view is broadcast
        document.querySelectorAll('.player-controls button')[0].click();
        
        window.scrollTo({ top: 0, behavior: 'smooth' });
    },

    closeGame() {
        document.getElementById('game-view').classList.remove('active');
        document.getElementById('season-view').classList.add('active');
        document.querySelectorAll('.nav-links a')[0].classList.add('active');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
};

document.addEventListener('DOMContentLoaded', () => app.init());
