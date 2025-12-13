// Check if the browser supports service workers and notifications
if ('serviceWorker' in navigator && 'PushManager' in window) {
    // Register service worker
    navigator.serviceWorker.register('/sw.js')
        .then(registration => {
            console.log('ServiceWorker registration successful');
            return registration.pushManager.getSubscription()
                .then(subscription => {
                    // If already subscribed, no need to ask again
                    if (subscription) {
                        console.log('User is already subscribed to push notifications');
                        return subscription;
                    }
                    // Otherwise, request permission
                    return requestNotificationPermission(registration);
                });
        })
        .catch(error => {
            console.error('ServiceWorker registration failed:', error);
        });
} else {
    console.warn('Push notifications are not supported in this browser');
}

// Function to request notification permission
function requestNotificationPermission(registration) {
    return new Promise((resolve, reject) => {
        const permissionResult = Notification.requestPermission(result => {
            if (result !== 'granted') {
                console.log('Notification permission not granted');
                reject(new Error('Permission not granted'));
                return;
            }
            
            // Get the server's public key
            fetch('/api/notifications/public-key')
                .then(response => response.json())
                .then(serverKey => {
                    // Subscribe to push notifications
                    return registration.pushManager.subscribe({
                        userVisibleOnly: true,
                        applicationServerKey: urlBase64ToUint8Array(serverKey.publicKey)
                    });
                })
                .then(subscription => {
                    // Send subscription to server
                    return fetch('/api/notifications/subscribe', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(subscription)
                    });
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Failed to save subscription');
                    }
                    console.log('Successfully subscribed to push notifications');
                    resolve();
                })
                .catch(error => {
                    console.error('Error subscribing to push notifications:', error);
                    reject(error);
                });
        });
    });
}

// Helper function to convert base64 string to Uint8Array
function urlBase64ToUint8Array(base64String) {
    const padding = '='.repeat((4 - base64String.length % 4) % 4);
    const base64 = (base64String + padding)
        .replace(/\-/g, '+')
        .replace(/_/g, '/');
    
    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);
    
    for (let i = 0; i < rawData.length; ++i) {
        outputArray[i] = rawData.charCodeAt(i);
    }
    return outputArray;
}

// Function to send a test notification
export function sendTestNotification() {
    fetch('/api/notifications/test', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log('Test notification sent:', data);
    })
    .catch(error => {
        console.error('Error sending test notification:', error);
    });
}

// Function to check notification permission status
export function checkNotificationPermission() {
    return Notification.permission;
}

// Function to subscribe to energy alerts
export function subscribeToEnergyAlerts(deviceId) {
    return fetch('/api/notifications/subscribe/energy-alert', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ deviceId })
    })
    .then(response => response.json());
}
