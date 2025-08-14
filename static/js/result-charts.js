// Initialize charts when DOM is ready (sentiment removed)
document.addEventListener('DOMContentLoaded', function () {
  // Helper: decode HTML entities for display labels
  function decodeHtmlEntities(text) {
    if (typeof text !== 'string') return text;
    var el = document.createElement('div');
    el.innerHTML = text;
    return el.textContent || el.innerText || text;
  }

  // Pull data injected by template
  var articles1 = Array.isArray(window.articles1) ? window.articles1 : [];
  var articles2 = Array.isArray(window.articles2) ? window.articles2 : null;
  var analysis1 = window.analysis1 || {};
  var analysis2 = window.analysis2 || null;
  var query1 = window.query1 || '';
  var query2 = window.query2 || '';

  // Timeline chart (daily article counts)
  if (analysis1.timeline && analysis1.timeline.length > 0) {
    var timelineCtx = document.getElementById('timelineChart').getContext('2d');

    var datasets = [{
      label: query1,
      data: (analysis1.timeline || []).map(item => ({ x: item.date, y: item.count })),
      borderColor: '#005e30',
      backgroundColor: 'rgba(0, 94, 48, 0.1)',
      borderWidth: 2,
      tension: 0.1,
      fill: true
    }];

    if (analysis2 && analysis2.timeline && analysis2.timeline.length > 0) {
      datasets.push({
        label: query2,
        data: (analysis2.timeline || []).map(item => ({ x: item.date, y: item.count })),
        borderColor: '#00a651',
        backgroundColor: 'rgba(0, 166, 81, 0.1)',
        borderWidth: 2,
        tension: 0.1,
        fill: true
      });
    }

    new Chart(timelineCtx, {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: 'time',
            time: { unit: 'day', displayFormats: { day: 'MMM d' } },
            title: { display: true, text: 'Date' }
          },
          y: {
            beginAtZero: true,
            title: { display: true, text: 'Number of Articles' },
            ticks: { precision: 0 }
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              title: function (context) {
                const date = new Date(context[0].parsed.x);
                return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
              }
            }
          }
        }
      }
    });
  }

  // Top sources horizontal bar (counts only)
  if (analysis1.sources && analysis1.sources.length > 0) {
    var sourcesCtx = document.getElementById('sourcesChart').getContext('2d');

    var topSources1 = (analysis1.sources || []).slice(0, 10);
    var labels = topSources1.map(s => decodeHtmlEntities(s.name));
    var datasets;

    if (analysis2 && analysis2.sources && analysis2.sources.length > 0) {
      var topSources2 = (analysis2.sources || []).slice(0, 10);

      var allSources = Array.from(new Set([
        ...topSources1.map(s => decodeHtmlEntities(s.name)),
        ...topSources2.map(s => decodeHtmlEntities(s.name))
      ]));

      labels = allSources;

      var data1 = labels.map(label => {
        const s = topSources1.find(x => decodeHtmlEntities(x.name) === label);
        return s ? s.count : 0;
      });

      var data2 = labels.map(label => {
        const s = topSources2.find(x => decodeHtmlEntities(x.name) === label);
        return s ? s.count : 0;
      });

      datasets = [
        {
          label: query1,
          data: data1,
          backgroundColor: 'rgba(0, 94, 48, 0.7)',
          borderColor: 'rgba(0, 94, 48, 1)',
          borderWidth: 1
        },
        {
          label: query2,
          data: data2,
          backgroundColor: 'rgba(0, 166, 81, 0.7)',
          borderColor: 'rgba(0, 166, 81, 1)',
          borderWidth: 1
        }
      ];
    } else {
      datasets = [{
        label: query1,
        data: topSources1.map(s => s.count),
        backgroundColor: 'rgba(0, 94, 48, 0.7)',
        borderColor: 'rgba(0, 94, 48, 1)',
        borderWidth: 1
      }];
    }

    new Chart(sourcesCtx, {
      type: 'bar',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        scales: {
          x: {
            beginAtZero: true,
            title: { display: true, text: 'Number of Articles' },
            ticks: { precision: 0 }
          },
          y: {
            ticks: {
              callback: function (value, index) {
                return decodeHtmlEntities(this.getLabelForValue(index));
              }
            }
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              title: function (items) {
                return decodeHtmlEntities(items[0].label);
              }
            }
          }
        }
      }
    });
  }

  // Copy to Claude button (sentiment removed)
  var copyBtn = document.getElementById('copyToClaudeBtn');
  if (copyBtn) {
    copyBtn.addEventListener('click', function () {
      var content = '';

      content += `Media Analysis for "${query1}"`;
      if (query2) content += ` vs "${query2}"`;
      content += '\n\n';

      if (analysis1 && analysis1.date_range) {
        content += `Date Range: ${analysis1.date_range.start} to ${analysis1.date_range.end}\n\n`;
      }

      content += '## Coverage Metrics\n\n';
      content += `${query1}: ${analysis1.total_articles || 0} articles\n`;
      if (analysis2) {
        content += `${query2}: ${(analysis2.total_articles || 0)} articles\n`;
      }
      content += '\n';

      content += '## Top Sources\n\n';
      if (analysis1 && analysis1.sources) {
        content += `${query1}: `;
        content += (analysis1.sources.slice(0, 5) || []).map(s => `${s.name} (${s.count})`).join(', ');
        content += '\n';
      }
      if (analysis2 && analysis2.sources) {
        content += `${query2}: `;
        content += (analysis2.sources.slice(0, 5) || []).map(s => `${s.name} (${s.count})`).join(', ');
        content += '\n';
      }
      content += '\n';

      content += '## Top Topics\n\n';
      if (analysis1 && analysis1.topics) {
        content += `${query1}: `;
        content += (analysis1.topics.slice(0, 10) || []).map(t => `${t.topic} (${t.count})`).join(', ');
        content += '\n';
      }
      if (analysis2 && analysis2.topics) {
        content += `${query2}: `;
        content += (analysis2.topics.slice(0, 10) || []).map(t => `${t.topic} (${t.count})`).join(', ');
        content += '\n';
      }
      content += '\n';

      content += '## Articles\n\n';
      content += `### ${query1} Articles\n\n`;
      (articles1.slice(0, 10) || []).forEach(a => {
        content += `- **${a.title}** (${a.source && a.source.name ? a.source.name : ''})\n`;
        content += `  Published: ${new Date(a.publishedAt).toLocaleDateString()}\n`;
        content += `  URL: ${a.url}\n\n`;
      });
      if (articles2 && articles2.length > 0) {
        content += `### ${query2} Articles\n\n`;
        (articles2.slice(0, 10) || []).forEach(a => {
          content += `- **${a.title}** (${a.source && a.source.name ? a.source.name : ''})\n`;
          content += `  Published: ${new Date(a.publishedAt).toLocaleDateString()}\n`;
          content += `  URL: ${a.url}\n\n`;
        });
      }

      navigator.clipboard.writeText(content).then(function () {
        alert('Analysis copied to clipboard! You can now paste it into Claude.');
      }, function (err) {
        console.error('Could not copy text: ', err);
        alert('Failed to copy to clipboard. Please try again.');
      });
    });
  }

  // Share button functionality (unchanged)
  var shareBtn = document.getElementById('shareBtn');
  if (shareBtn) {
    shareBtn.addEventListener('click', function () {
      var url = window.location.href;
      navigator.clipboard.writeText(url).then(function () {
        alert('URL copied to clipboard! You can now share it with others.');
      }, function (err) {
        console.error('Could not copy text: ', err);
        alert('Failed to copy to clipboard. Please try again.');
      });
    });
  }
});
