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
- Sarı nokta ile bakış noktası gösterimi

### Mouse Control
- Smooth cursor movement
- Göz bebeği pozisyonuna dayalı hassas kontrol
- Adjustable sensitivity with slider
- Kalibrasyon sonrası otomatik aktivasyon
- Safety bounds kontrolü

### Calibration
- 9-nokta kalibrasyon sistemi
- Her nokta için veri toplama ve analiz
- Göz bebeğinin relatif pozisyonuna dayalı kalibrasyon
- Kafa pozisyonundan bağımsız kalibrasyon
- Re-calibration on demand

### User Interface
- Modern CustomTkinter-based UI
- Real-time webcam görüntüsü
- Basit ve anlaşılır kontroller
- Status indicators
- Smooth factor ayarı

## Technical Requirements
- Compatible with standard webcams (720p+)
- Screen resolution independence
- CPU usage optimization
- Robust error handling

## Settings & Customization
- Tracking sensitivity
- Mouse movement smoothing
- Camera selection
- Kalibrasyon hassasiyeti

## Completed Tasks
- [x] Basic webcam integration
- [x] Eye detection algorithm
- [x] Modern UI implementation
- [x] Kalibrasyon sistemi
- [x] Mouse control implementation
- [x] Göz bebeği pozisyon hesaplama
- [x] Aynalı görüntü desteği
- [x] Real-time visualization
- [x] Smoothing control
- [x] Multi-camera support
- [x] Error handling

## Tasks To Be Done
- [ ] Click actions via eye gestures
- [ ] Profile management
- [ ] Multi-monitor support
- [ ] Testing & validation
- [ ] Documentation
- [ ] Installer creation
- [ ] Performance optimization
- [ ] Cloud profile sync

## Success Metrics
- Tracking accuracy >95%
- Latency <50ms
- CPU usage <10%
- Calibration time <30s
- User satisfaction >4/5

## Timeline
- Phase 1: Core tracking ✓
- Phase 2: Mouse control ✓
- Phase 3: UI & Settings ✓
- Phase 4: Testing & Polish (In Progress)

## Future Enhancements
- AI-based accuracy improvements
- Custom gesture recognition
- Accessibility features
- Game-specific profiles
- Cloud profile sync
- Multiple monitor support
- Eye blink detection for clicks