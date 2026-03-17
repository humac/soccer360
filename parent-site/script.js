const LOCAL_STORAGE_KEY = 'soccer360_games';
const CONFIG_STORAGE_KEY = 'soccer360_config';

const app = {
    games: [],
    config: {
        dashboardUrl: `http://${window.location.hostname}:8088`,
        myTeam: 'OSU STEEL' // Default to search for
    },
    activeMatchId: null,
    activeSegmentId: null,

    init() {
        this.loadState();
        this.bindEvents();
        this.renderSeasonView();
        this.renderConfiguredGames();
        this.updateConfigUI();
    },

    loadState() {
        const storedGames = localStorage.getItem(LOCAL_STORAGE_KEY);
        if (storedGames) {
            try { this.games = JSON.parse(storedGames); } 
            catch (e) { console.error("Failed to parse stored games", e); }
        }

        const storedConfig = localStorage.getItem(CONFIG_STORAGE_KEY);
        if (storedConfig) {
            try { 
                const parsed = JSON.parse(storedConfig);
                this.config = { ...this.config, ...parsed };
            }
            catch (e) { console.error("Failed to parse config", e); }
        }
    },

    saveState() {
        localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(this.games));
        localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(this.config));
        this.renderSeasonView();
        this.renderConfiguredGames();
    },

    updateConfigUI() {
        const urlInput = document.getElementById('opt-dashboard-url');
        if(urlInput) {
            urlInput.value = this.config.dashboardUrl;
            urlInput.placeholder = `http://${window.location.hostname}:8088`;
        }
    },

    getMatchGroups() {
        const groups = {};
        
        this.games.forEach(game => {
            // Normalize title to find the base match
            // Remove "1st Half", "2nd Half", "Part X", "vX" etc.
            let baseTitle = game.title
                .replace(/\d+(st|nd|rd|th)\s+half/gi, '')
                .replace(/part\s+\d+/gi, '')
                .replace(/\(\d+\)/g, '')
                .trim();
            
            const key = baseTitle.toUpperCase();
            if (!groups[key]) {
                groups[key] = {
                    id: key,
                    title: baseTitle,
                    date: game.date,
                    segments: [],
                    score: game.score || ''
                };
            }
            
            groups[key].segments.push(game);
            if (game.score && !groups[key].score) groups[key].score = game.score;
            // Keep the earliest date if multiple segments exist
            if (new Date(game.date) < new Date(groups[key].date)) groups[key].date = game.date;
            // Sort segments alphabetically (so 1st Half < 2nd Half)
            groups[key].segments.sort((a, b) => {
                if (a.title < b.title) return -1;
                if (a.title > b.title) return 1;
                return 0;
            });
        });

        return Object.values(groups).sort((a,b) => new Date(b.date) - new Date(a.date));
    },

    calculateStats(matchGroups) {
        let played = matchGroups.length;
        let wins = 0;
        let draws = 0;

        matchGroups.forEach(m => {
            if (!m.score) return;
            const parts = m.score.match(/(\d+)\s*[-:]\s*(\d+)/);
            if (parts) {
                const s1 = parseInt(parts[1]);
                const s2 = parseInt(parts[2]);
                if (s1 > s2) wins++;
                else if (s1 === s2) draws++;
            }
        });

        document.getElementById('stat-played').textContent = played;
        document.getElementById('stat-wins').textContent = wins;
        document.getElementById('stat-draws').textContent = draws;
    },

    bindEvents() {
        // Nav links
        document.querySelectorAll('.nav-links a').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                document.querySelectorAll('.nav-links a').forEach(l => l.classList.remove('active'));
                e.target.classList.add('active');
                
                const view = e.target.dataset.view;
                document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
                
                if (view === 'season') {
                    document.getElementById('season-view').classList.add('active');
                    this.renderSeasonView();
                } else if (view === 'options') {
                    document.getElementById('options-view').classList.add('active');
                } else {
                    document.getElementById('season-view').classList.add('active');
                }
            });
        });

        // Config updates
        document.getElementById('opt-dashboard-url')?.addEventListener('change', (e) => {
            this.config.dashboardUrl = e.target.value.replace(/\/$/, '');
            this.saveState();
        });

        // Player controls
        document.querySelectorAll('.view-toggle button').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.view-toggle button').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                
                const isTactical = e.target.textContent.toLowerCase().includes('tactical');
                this.switchVideoView(isTactical ? 'tactical' : 'broadcast');
            });
        });

        // Options form
        document.getElementById('add-game-form')?.addEventListener('submit', (e) => {
            e.preventDefault();
            
            let highlights = [];
            const rawH = document.getElementById('opt-highlights').value;
            if (rawH) {
                try {
                    const parsed = JSON.parse(rawH);
                    highlights = parsed.clips || (Array.isArray(parsed) ? parsed : []);
                } catch (err) {
                    alert("Warning: Could not parse highlights JSON.");
                }
            }

            const editId = document.getElementById('opt-edit-id').value;
            const gameData = {
                id: editId || ('game-' + Date.now()),
                title: document.getElementById('opt-title').value,
                date: document.getElementById('opt-date').value,
                score: document.getElementById('opt-score').value,
                basePath: document.getElementById('opt-path').value.replace(/\/$/, ''),
                matchName: document.getElementById('opt-title').value,
                mode: document.getElementById('opt-mode').value,
                highlights: highlights
            };

            if (editId) {
                // Determine if this is a group ID or a single game ID
                const isGroup = this.games.filter(g => g.id === editId).length === 0; // if not found directly, assume group ID
                if (isGroup) {
                    const groupTitle = document.getElementById('opt-title').value;
                    const groupDate = document.getElementById('opt-date').value;
                    const groupScore = document.getElementById('opt-score').value;

                    this.games.forEach(g => {
                        const baseTitle = g.title
                            .replace(/\d+(st|nd|rd|th)\s+half/gi, '')
                            .replace(/part\s+\d+/gi, '')
                            .replace(/\(\d+\)/g, '')
                            .trim()
                            .toUpperCase();
                        
                        if (baseTitle === editId) { // editId holds the group key
                            // We preserve the specific segment title suffix (e.g. " 1st Half")
                            const suffixMatch = g.title.match(/(\s+\d+(st|nd|rd|th)\s+half|\s+part\s+\d+|\s+\(\d+\))/i);
                            const suffix = suffixMatch ? suffixMatch[0] : '';
                            g.title = groupTitle + suffix;
                            g.matchName = g.title;

                            g.date = groupDate;
                            g.score = groupScore;
                            
                            // If they provided highlights on a group edit, we apply to all segments
                            if (highlights.length > 0) g.highlights = highlights;
                        }
                    });
                } else {
                    const idx = this.games.findIndex(g => g.id === editId);
                    if (idx !== -1) {
                        gameData.id = this.games[idx].id; // preserve ID
                        this.games[idx] = gameData;
                    }
                }
                this.cancelEdit();
            } else {
                this.addGame(gameData);
            }
            
            this.saveState();
            e.target.reset();
        });

        document.getElementById('opt-cancel-edit')?.addEventListener('click', () => {
            this.cancelEdit();
        });

        document.getElementById('clear-games-btn')?.addEventListener('click', () => {
            if(confirm('Are you sure you want to clear all configured games?')) {
                this.games = [];
                this.saveState();
            }
        });

        document.getElementById('load-auto-btn')?.addEventListener('click', () => {
            this.autoDetectMatches();
        });
    },

    async autoDetectMatches() {
        const btn = document.getElementById('load-auto-btn');
        const originalText = btn.textContent;
        btn.textContent = 'Detecting...';
        btn.disabled = true;

        try {
            const resp = await fetch(`${this.config.dashboardUrl}/api/media/matches`);
            if(!resp.ok) throw new Error("Dashboard not reachable");
            const matches = await resp.json();
            
            if (matches.length === 0) {
                alert("No matches found in dashboard.");
            } else {
                let updatedCount = 0;
                let newCount = 0;
                
                for (const m of matches) {
                    const existingGame = this.games.find(g => g.title === m.name);
                    
                    if (existingGame) {
                        // It exists, check if we need to fetch missing highlights
                        if (!existingGame.highlights || existingGame.highlights.length === 0) {
                            const highlights = await this.fetchHighlights(m.name);
                            if (highlights.length > 0) {
                                existingGame.highlights = highlights;
                                updatedCount++;
                            }
                        }
                    } else {
                        // New game
                        const highlights = await this.fetchHighlights(m.name);
                        this.addGame({
                            id: 'game-' + Date.now() + Math.random(),
                            title: m.name,
                            date: m.processed_at && m.processed_at !== '--' ? new Date(m.processed_at).toLocaleDateString() : new Date().toLocaleDateString(),
                            score: '',
                            matchName: m.name,
                            mode: m.mode || 'normal',
                            highlights: highlights
                        }, false);
                        newCount++;
                    }
                }
                this.saveState();
                alert(`Detected ${newCount} new recordings. Fetched missing highlights for ${updatedCount} existing recordings.`);
            }
        } catch (err) {
            alert(`Failed to detect matches: ${err.message}`);
        } finally {
            btn.textContent = originalText;
            btn.disabled = false;
        }
    },

    async fetchHighlights(matchName) {
        try {
            const resp = await fetch(`${this.config.dashboardUrl}/api/media/${matchName}/highlights/highlights.json`);
            if (resp.ok) {
                const data = await resp.json();
                return data.clips || (Array.isArray(data) ? data : []);
            }
        } catch (e) {
            console.warn(`Could not fetch highlights for ${matchName}`, e);
        }
        return [];
    },

    addGame(gameConfig, save = true) {
        this.games.push(gameConfig);
        if(save) {
            this.saveState();
            alert('Game added successfully!');
        }
    },

    editGame(idOrKey) {
        let title, date, score, basePath, mode, hls;

        const groupMatch = this.getMatchGroups().find(g => g.id === idOrKey);
        if (groupMatch) {
            // Group Edit
            title = groupMatch.title;
            date = groupMatch.date;
            score = groupMatch.score;
            basePath = 'Multiple Paths (Editing Group)';
            mode = groupMatch.segments[0].mode;
            hls = groupMatch.segments[0].highlights || [];
            document.getElementById('opt-path').disabled = true;
        } else {
            // Single Segment Edit (Options views configured games)
            const game = this.games.find(g => g.id === idOrKey);
            if (!game) return;
            title = game.title;
            date = game.date;
            score = game.score;
            basePath = game.basePath;
            mode = game.mode;
            hls = game.highlights || [];
            document.getElementById('opt-path').disabled = false;
        }

        document.getElementById('opt-edit-id').value = idOrKey;
        document.getElementById('opt-title').value = title || '';
        document.getElementById('opt-date').value = date || '';
        document.getElementById('opt-score').value = score || '';
        document.getElementById('opt-path').value = basePath || '';
        document.getElementById('opt-mode').value = mode || 'normal';
        document.getElementById('opt-highlights').value = JSON.stringify({ clips: hls }, null, 2);

        document.getElementById('opt-submit-btn').textContent = 'UPDATE MATCH';
        document.getElementById('opt-cancel-edit').style.display = 'inline-block';
        window.scrollTo({ top: 0, behavior: 'smooth' });
    },

    cancelEdit() {
        document.getElementById('opt-edit-id').value = '';
        document.getElementById('add-game-form').reset();
        document.getElementById('opt-submit-btn').textContent = 'ADD MATCH';
        document.getElementById('opt-cancel-edit').style.display = 'none';
        document.getElementById('opt-path').disabled = false;
    },

    removeGame(id) {
        if(confirm('Remove this match recording?')) {
            this.games = this.games.filter(g => g.id !== id);
            this.saveState();
        }
    },

    renderConfiguredGames() {
        const list = document.getElementById('configured-games-list');
        if (!list) return;

        list.innerHTML = '';
        if (this.games.length === 0) {
            list.innerHTML = '<p style="color:var(--text-muted);font-size:0.9rem;">No games configured yet.</p>';
            return;
        }

        this.games.forEach(game => {
            const item = document.createElement('div');
            item.className = 'config-game-item';
            item.innerHTML = `
                <div class="config-game-info">
                    <strong>${game.title}</strong>
                    <span>${game.date} | ${game.highlights.length} Highlights</span>
                </div>
                <div class="config-game-actions">
                    <button class="btn-secondary" style="padding:0.4rem 0.8rem; font-size:0.8rem;" onclick="app.editGame('${game.id}')">Edit</button>
                    <button class="btn-danger" style="padding:0.4rem 0.8rem; font-size:0.8rem;" onclick="app.removeGame('${game.id}')">Remove</button>
                </div>
            `;
            list.appendChild(item);
        });
    },

    renderSeasonView() {
        const grid = document.getElementById('matches-grid');
        if (!grid) return;

        const matchGroups = this.getMatchGroups();
        this.calculateStats(matchGroups);

        grid.innerHTML = '';
        if (matchGroups.length === 0) {
            grid.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 2rem;">No matches found.</p>';
            return;
        }

        matchGroups.forEach(m => {
            const titleMatch = m.title.match(/(.*)\sVS\s(.*)/i);
            const titleHtml = titleMatch ? `<h2>${titleMatch[1].trim()} <span>VS</span> ${titleMatch[2].trim()}</h2>` : `<h2>${m.title}</h2>`;

            const card = document.createElement('div');
            card.className = 'match-card';
            card.onclick = () => this.openMatch(m.id);
            card.innerHTML = `
                <div class="card-bg"></div>
                <div class="match-date">${m.date}</div>
                <div class="match-teams">${titleHtml}</div>
                <div class="match-meta">
                    ${m.score ? `<span class="badge score">${m.score}</span>` : ''}
                    <span class="badge mode">${m.segments.length} RECORDING${m.segments.length > 1 ? 'S' : ''}</span>
                </div>
                <div class="hover-reveal">
                    VIEW MATCH <span>&rarr;</span>
                </div>
                <button class="match-card-edit-btn" onclick="app.triggerEdit(event, '${m.id}')">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>
                </button>
            `;
            grid.appendChild(card);
        });
    },

    triggerEdit(event, gameId) {
        // Prevent opening the match card
        event.stopPropagation();
        
        // Navigate to options view
        document.querySelectorAll('.nav-links a').forEach(l => l.classList.remove('active'));
        document.querySelector('.nav-links a[data-view="options"]').classList.add('active');
        document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
        document.getElementById('options-view').classList.add('active');
        
        // Trigger edit
        this.editGame(gameId);
    },

    openMatch(matchId) {
        const groups = this.getMatchGroups();
        const match = groups.find(m => m.id === matchId);
        if (!match) return;

        this.activeMatchId = matchId;
        
        // Setup UI
        const titleEl = document.getElementById('active-game-title');
        titleEl.textContent = match.title;
        document.getElementById('active-game-score').textContent = match.score || '-';
        document.getElementById('active-game-date').textContent = match.date;

        // Add Edit button to header
        let editBtn = document.getElementById('game-view-edit-btn');
        if (!editBtn) {
            editBtn = document.createElement('button');
            editBtn.id = 'game-view-edit-btn';
            editBtn.className = 'btn-secondary';
            editBtn.style.marginLeft = '1rem';
            editBtn.style.padding = '0.3rem 0.6rem';
            editBtn.style.fontSize = '0.8rem';
            editBtn.innerHTML = 'Edit Match';
        }
        titleEl.appendChild(editBtn);
        editBtn.onclick = () => this.triggerEdit(new Event('click'), match.id); // pass Group ID

        document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));

        document.getElementById('game-view').classList.add('active');
        
        // Render Segment Selector if multiple
        const segSelector = document.getElementById('segment-selector');
        segSelector.innerHTML = '';
        if (match.segments.length > 1) {
            segSelector.style.display = 'flex';
            match.segments.forEach(seg => {
                const btn = document.createElement('button');
                btn.className = 'segment-btn';
                btn.textContent = seg.title.replace(match.title, '').trim() || 'Part';
                btn.onclick = () => this.selectSegment(seg.id);
                btn.dataset.id = seg.id;
                segSelector.appendChild(btn);
            });
        } else {
            segSelector.style.display = 'none';
        }

        // Load first segment
        this.selectSegment(match.segments[0].id);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    },

    selectSegment(segmentId) {
        const game = this.games.find(g => g.id === segmentId);
        if (!game) return;

        this.activeSegmentId = segmentId;

        // Update segment active states
        document.querySelectorAll('.segment-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.id === segmentId);
        });

        // Setup Video paths
        const videoEl = document.getElementById('game-video');
        const matchName = game.matchName || game.title;
        
        videoEl.dataset.broadcastSrc = `${this.config.dashboardUrl}/api/media/${matchName}/broadcast.mp4`;
        videoEl.dataset.tacticalSrc = `${this.config.dashboardUrl}/api/media/${matchName}/tactical_wide.mp4`;
        
        // Reset to broadcast and play
        document.querySelectorAll('.view-toggle button')[0].click();
        
        this.renderHighlights(game);
        this.updateDownloads(game);
    },

    renderHighlights(game) {
        const list = document.getElementById('highlights-list');
        if (!list) return;

        list.innerHTML = '';
        if (!game.highlights || game.highlights.length === 0) {
            list.innerHTML = '<p style="color:var(--text-muted);font-size:0.9rem;">No highlights for this segment.</p>';
            return;
        }

        game.highlights.forEach(h => {
            const item = document.createElement('div');
            item.className = 'highlight-item';
            const minutes = Math.floor(h.start_sec / 60);
            const seconds = Math.floor(h.start_sec % 60).toString().padStart(2, '0');
            const eventDesc = h.event_types ? h.event_types.join(', ').replace(/_/g, ' ') : 'Highlight';

            item.innerHTML = `
                <div class="h-time">${minutes}:${seconds}</div>
                <div class="h-desc">${eventDesc}</div>
                <button class="action-btn play-h">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M5 3l14 9-14 9V3z"/></svg>
                </button>
            `;
            item.onclick = () => this.seekTo(h.start_sec);
            list.appendChild(item);
        });
    },

    seekTo(seconds) {
        const videoEl = document.getElementById('game-video');
        if (videoEl) {
            const broadcastBtn = document.querySelectorAll('.view-toggle button')[0];
            if (!broadcastBtn.classList.contains('active')) broadcastBtn.click();
            videoEl.currentTime = seconds;
            videoEl.play().catch(e => console.log('Autoplay blocked', e));
        }
    },

    switchVideoView(type) {
        const videoEl = document.getElementById('game-video');
        const placeholder = document.getElementById('video-placeholder');
        const src = type === 'tactical' ? videoEl.dataset.tacticalSrc : videoEl.dataset.broadcastSrc;
        
        if (src) {
            const currentTime = videoEl.currentTime;
            const wasPlaying = !videoEl.paused;
            videoEl.src = src;
            videoEl.style.display = 'block';
            placeholder.style.display = 'none';
            
            videoEl.onloadedmetadata = () => {
                videoEl.currentTime = currentTime;
                if(wasPlaying) videoEl.play().catch(e => {});
                videoEl.onloadedmetadata = null;
            };
        }
    },

    updateDownloads(game) {
        const matchName = game.matchName || game.title;
        const base = `${this.config.dashboardUrl}/api/media/${matchName}`;
        const dLinks = document.querySelectorAll('.download-card');
        if(dLinks.length >= 3) {
            dLinks[0].href = `${base}/broadcast.mp4`;
            dLinks[1].href = `${base}/tactical_wide.mp4`;
            dLinks[2].href = `${base}/detections.jsonl`;
        }
    },

    closeGame() {
        this.activeMatchId = null;
        this.activeSegmentId = null;
        const videoEl = document.getElementById('game-video');
        if(videoEl) { videoEl.pause(); videoEl.src = ''; }
        document.getElementById('game-view').classList.remove('active');
        document.getElementById('season-view').classList.add('active');
        document.querySelectorAll('.nav-links a').forEach(l => l.classList.remove('active'));
        document.querySelectorAll('.nav-links a')[0].classList.add('active');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
};

document.addEventListener('DOMContentLoaded', () => app.init());
