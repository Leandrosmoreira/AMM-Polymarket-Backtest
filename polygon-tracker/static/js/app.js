/**
 * Gabagool Tracker - Frontend JavaScript
 */

// State
let allTransactions = [];
let filteredTransactions = [];
let currentPage = 1;
const perPage = 25;

// DOM Elements
const syncBtn = document.getElementById('syncBtn');
const txBody = document.getElementById('txBody');
const lastSyncEl = document.getElementById('lastSync');
const filterSide = document.getElementById('filterSide');
const filterToken = document.getElementById('filterToken');
const filterDateFrom = document.getElementById('filterDateFrom');
const filterDateTo = document.getElementById('filterDateTo');
const clearFiltersBtn = document.getElementById('clearFilters');
const prevPageBtn = document.getElementById('prevPage');
const nextPageBtn = document.getElementById('nextPage');
const pageInfo = document.getElementById('pageInfo');

// Stats elements
const totalTxEl = document.getElementById('totalTx');
const totalBuysEl = document.getElementById('totalBuys');
const totalSellsEl = document.getElementById('totalSells');
const uniqueMarketsEl = document.getElementById('uniqueMarkets');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadTransactions();
    setupEventListeners();
});

function setupEventListeners() {
    syncBtn.addEventListener('click', syncTransactions);
    filterSide.addEventListener('change', applyFilters);
    filterToken.addEventListener('change', applyFilters);
    filterDateFrom.addEventListener('change', applyFilters);
    filterDateTo.addEventListener('change', applyFilters);
    clearFiltersBtn.addEventListener('click', clearFilters);
    prevPageBtn.addEventListener('click', () => changePage(-1));
    nextPageBtn.addEventListener('click', () => changePage(1));
}

async function loadTransactions() {
    try {
        showLoading();
        const response = await fetch('/api/transactions');
        const data = await response.json();

        if (data.success) {
            allTransactions = data.transactions;
            filteredTransactions = [...allTransactions];
            updateStats(data.stats);
            updateLastSync(data.last_sync);
            renderTransactions();
        } else {
            showError(data.error || 'Failed to load transactions');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    }
}

async function syncTransactions() {
    try {
        syncBtn.disabled = true;
        syncBtn.classList.add('syncing');
        syncBtn.innerHTML = '<span class="sync-icon">&#x21bb;</span> Syncing...';

        const response = await fetch('/api/sync', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            allTransactions = data.transactions;
            applyFilters();
            updateStats(data.stats);
            updateLastSync(data.last_sync);
            showToast(data.message, 'success');
        } else {
            showError(data.error || 'Sync failed');
        }
    } catch (error) {
        showError('Sync error: ' + error.message);
    } finally {
        syncBtn.disabled = false;
        syncBtn.classList.remove('syncing');
        syncBtn.innerHTML = '<span class="sync-icon">&#x21bb;</span> SYNC';
    }
}

function applyFilters() {
    const side = filterSide.value;
    const token = filterToken.value;
    const dateFrom = filterDateFrom.value;
    const dateTo = filterDateTo.value;

    filteredTransactions = allTransactions.filter(tx => {
        // Side filter
        if (side !== 'all' && tx.side !== side) return false;

        // Token filter
        if (token !== 'all' && tx.token_type !== token) return false;

        // Date from filter
        if (dateFrom && tx.date < dateFrom) return false;

        // Date to filter
        if (dateTo && tx.date > dateTo) return false;

        return true;
    });

    currentPage = 1;
    renderTransactions();
    updateFilteredStats();
}

function clearFilters() {
    filterSide.value = 'all';
    filterToken.value = 'all';
    filterDateFrom.value = '';
    filterDateTo.value = '';
    filteredTransactions = [...allTransactions];
    currentPage = 1;
    renderTransactions();
    // Restore original stats
    if (allTransactions.length > 0) {
        updateFilteredStats();
    }
}

function updateFilteredStats() {
    const txs = filteredTransactions;
    totalTxEl.textContent = txs.length;
    totalBuysEl.textContent = txs.filter(tx => tx.side === 'BUY').length;
    totalSellsEl.textContent = txs.filter(tx => tx.side === 'SELL').length;
    uniqueMarketsEl.textContent = new Set(txs.map(tx => tx.contract)).size;
}

function updateStats(stats) {
    totalTxEl.textContent = stats.total_transactions || 0;
    totalBuysEl.textContent = stats.total_buys || 0;
    totalSellsEl.textContent = stats.total_sells || 0;
    uniqueMarketsEl.textContent = stats.unique_markets || 0;
}

function updateLastSync(timestamp) {
    if (timestamp) {
        const date = new Date(timestamp);
        lastSyncEl.textContent = `Last sync: ${date.toLocaleString()}`;
    }
}

function renderTransactions() {
    const start = (currentPage - 1) * perPage;
    const end = start + perPage;
    const pageTxs = filteredTransactions.slice(start, end);

    if (pageTxs.length === 0) {
        txBody.innerHTML = `
            <tr>
                <td colspan="6">
                    <div class="no-data">
                        <div class="no-data-icon">&#x1F50D;</div>
                        <p>No transactions found</p>
                    </div>
                </td>
            </tr>
        `;
        updatePagination();
        return;
    }

    txBody.innerHTML = pageTxs.map(tx => `
        <tr>
            <td class="datetime">
                <span class="date">${tx.date}</span>
                <span class="time">${tx.time}</span>
            </td>
            <td>
                <span class="side-tag ${tx.side.toLowerCase()}">${tx.side}</span>
            </td>
            <td>
                <span class="token-tag ${tx.token_type.toLowerCase()}">${tx.token_type}</span>
            </td>
            <td class="amount">
                ${formatAmount(tx.amount)}
            </td>
            <td>
                ${tx.method ? `<span class="method-tag">${tx.method}</span>` : '-'}
            </td>
            <td>
                <a href="https://polygonscan.com/tx/${tx.hash}" target="_blank" class="hash-link">
                    ${tx.hash.slice(0, 8)}...${tx.hash.slice(-6)}
                </a>
            </td>
        </tr>
    `).join('');

    updatePagination();
}

function formatAmount(amount) {
    if (!amount || amount === 0) return '-';
    if (amount >= 1000000) {
        return (amount / 1000000).toFixed(2) + 'M';
    }
    if (amount >= 1000) {
        return (amount / 1000).toFixed(2) + 'K';
    }
    return amount.toFixed(2);
}

function updatePagination() {
    const totalPages = Math.ceil(filteredTransactions.length / perPage);
    pageInfo.textContent = `Page ${currentPage} of ${totalPages || 1}`;
    prevPageBtn.disabled = currentPage <= 1;
    nextPageBtn.disabled = currentPage >= totalPages;
}

function changePage(delta) {
    const totalPages = Math.ceil(filteredTransactions.length / perPage);
    const newPage = currentPage + delta;

    if (newPage >= 1 && newPage <= totalPages) {
        currentPage = newPage;
        renderTransactions();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

function showLoading() {
    txBody.innerHTML = `
        <tr class="loading-row">
            <td colspan="6">
                <div class="loading">
                    <div class="spinner"></div>
                    <span>Loading transactions...</span>
                </div>
            </td>
        </tr>
    `;
}

function showError(message) {
    showToast(message, 'error');
    txBody.innerHTML = `
        <tr>
            <td colspan="6">
                <div class="no-data">
                    <div class="no-data-icon">&#x26A0;</div>
                    <p>Error: ${message}</p>
                    <button class="btn btn-primary" onclick="loadTransactions()" style="margin-top: 16px;">
                        Retry
                    </button>
                </div>
            </td>
        </tr>
    `;
}

function showToast(message, type = 'success') {
    // Remove existing toasts
    const existingToast = document.querySelector('.toast');
    if (existingToast) {
        existingToast.remove();
    }

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 4000);
}
