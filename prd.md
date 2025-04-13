# Eye Tracking Mouse Control - Product Requirements Document

## Overview
A webcam-based eye tracking solution that enables hands-free mouse control through precise eye movement detection and tracking.

## Core Features

### Eye Detection & Tracking
- Real-time göz bebeği (iris) detection using MediaPipe
- Kafa pozisyonundan bağımsız göz takibi
- Göz çukuru içindeki göz bebeği pozisyonunun relatif hesaplanması
- Ekran koordinatlarına doğru dönüşüm
- Low latency tracking (<50ms)
- Aynalı görüntü desteği
- Debug panelinde bakış noktası gösterimi

### Mouse Control
- Smooth cursor movement
- Göz bebeği pozisyonuna dayalı hassas kontrol
- Adjustable sensitivity with slider
- Kalibrasyon sonrası otomatik aktivasyon
- Safety bounds kontrolü
- ESC + L ile mouse kontrolünü devre dışı bırakma

### Calibration System
- 3 aşamalı kalibrasyon (yakın/orta/uzak)
- Her aşamada 5 nokta kalibrasyonu (4 köşe + merkez)
- Her nokta için 2 saniye veri toplama
- Kalibrasyon profillerini kaydetme/yükleme
- Re-calibration on demand

### User Interface
- Modern CustomTkinter-based UI
- Real-time webcam görüntüsü
- Basit ve anlaşılır kontroller
- Status indicators
- Debug visualization panel
- Smooth factor ayarı

## Technical Requirements
- Compatible with standard webcams (720p+)
- Screen resolution independence
- CPU usage optimization
- Robust error handling
- Profile data persistence

## Settings & Customization
- Tracking sensitivity
- Mouse movement smoothing
- Camera selection
- Kalibrasyon profil yönetimi

## Completed Tasks
- [x] Basic webcam integration
- [x] Eye detection algorithm
- [x] Modern UI implementation
- [x] 3-aşamalı kalibrasyon sistemi
- [x] Mouse control implementation
- [x] Göz bebeği pozisyon hesaplama
- [x] Aynalı görüntü desteği
- [x] Real-time visualization
- [x] Smoothing control
- [x] Multi-camera support
- [x] Error handling
- [x] Debug visualization panel
- [x] Kalibrasyon profil sistemi
- [x] Algoritma optimizasyonu (3D göz hareketi analizi)

## Tasks To Be Done
- [ ] Click actions via eye gestures
- [ ] Multi-monitor support
- [ ] Documentation
- [ ] Installer creation
- [ ] Performance optimization
- [ ] Eye blink detection
- [ ] Game-specific profile templates
- [ ] Advanced accessibility features
- [ ] Profile import/export functionality
- [ ] User tutorial/onboarding system

## Success Metrics
- Tracking accuracy >95%
- Latency <50ms
- CPU usage <10%
- Calibration time <60s
- User satisfaction >4/5

## Timeline
- Phase 1: Core tracking ✓
- Phase 2: Mouse control ✓
- Phase 3: UI & Settings ✓
- Phase 4: Profile System & Optimization (In Progress)

## Future Enhancements
- AI-based accuracy improvements
- Custom gesture recognition
- Accessibility features
- Game-specific profiles
- Multiple monitor support
- Eye blink detection for clicks