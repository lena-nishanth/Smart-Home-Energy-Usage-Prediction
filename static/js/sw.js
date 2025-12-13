// Service Worker for handling push notifications
self.addEventListener('push', function(event) {
    const data = event.data ? event.data.json() : {};
    const title = data.title || 'Energy Monitor Alert';
    const options = {
        body: data.body || 'You have a new notification',
        icon: '/static/images/icon-192x192.png',
        badge: '/static/images/badge-72x72.png',
        data: data.data || {},
        vibrate: [200, 100, 200, 100, 200, 100, 200],
        tag: 'energy-monitor-notification',
        renotify: true,
        actions: data.actions || []
    };

    event.waitUntil(
        self.registration.showNotification(title, options)
    );
});

self.addEventListener('notificationclick', function(event) {
    event.notification.close();
    
    // Handle the notification click
    if (event.notification.data && event.notification.data.url) {
        event.waitUntil(
            clients.openWindow(event.notification.data.url)
        );
    }
});
