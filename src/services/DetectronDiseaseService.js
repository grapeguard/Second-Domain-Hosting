// Enhanced Detectron2 Service for GrapeGuard with Visualization
// src/services/DetectronDiseaseService.js

class DetectronDiseaseService {
  constructor() {
    this.modelEndpoint = process.env.REACT_APP_DETECTRON_ENDPOINT || 'http://localhost:5000/predict';
    this.visualizeEndpoint = process.env.REACT_APP_DETECTRON_ENDPOINT || 'http://localhost:5000/visualize';
    this.fallbackService = null;
    this.isModelLoaded = false;
    this.modelVersion = '1.0.0';
    
    // Your exact disease mapping from training
    this.diseaseMapping = {
      1: {
        name: "Karpa (Anthracnose)",
        marathi: "कर्पा रोग",
        severity: "High",
        recommendations: [
          "Remove infected leaves immediately and burn them",
          "Spray Chlorothalonil (0.2%) or Carbendazim (0.1%) every 10-15 days",
          "Improve air circulation by pruning dense foliage",
          "Avoid overhead irrigation during humid weather",
          "Apply copper-based fungicide as preventive measure",
          "Ensure proper drainage to reduce humidity around plants"
        ]
      },
      2: {
        name: "Bhuri (Powdery Mildew)",
        marathi: "भुरी रोग", 
        severity: "Medium",
        recommendations: [
          "Apply sulfur dust (20-25 kg/ha) during early morning",
          "Spray Triadimefon (0.1%) or Penconazole (0.05%) every 7-10 days",
          "Improve canopy management for better air circulation",
          "Avoid excessive nitrogen fertilization",
          "Remove affected leaves and destroy them immediately",
          "Use Potassium Bicarbonate spray as organic option"
        ]
      },
      3: {
        name: "Bokadlela (Borer Infestation)",
        marathi: "बोकाडलेला",
        severity: "High",
        recommendations: [
          "Install pheromone traps (10-12 per acre) for monitoring",
          "Apply Spinosad (0.01%) or Chlorantraniliprole (0.006%)",
          "Prune and destroy affected branches immediately",
          "Apply neem oil spray (0.3%) every 15 days",
          "Monitor weekly for new entry holes",
          "Use biological control agents like Trichogramma wasps"
        ]
      },
      4: {
        name: "Davnya (Downy Mildew)",
        marathi: "दवयाचा रोग",
        severity: "High",
        recommendations: [
          "Spray Metalaxyl + Mancozeb (0.25%) immediately after rain",
          "Apply Copper oxychloride (0.3%) as preventive spray",
          "Ensure proper drainage and avoid waterlogging",
          "Avoid evening irrigation to reduce leaf wetness",
          "Prune canopy for better sunlight penetration",
          "Use Fosetyl Aluminum (0.3%) for systemic protection"
        ]
      },
      5: {
        name: "Healthy",
        marathi: "निरोगी पान",
        severity: "None",
        recommendations: [
          "Continue regular monitoring and inspection",
          "Maintain proper irrigation schedule",
          "Apply preventive fungicide spray monthly",
          "Ensure balanced nutrition (NPK + micronutrients)",
          "Keep vineyard clean of fallen leaves",
          "Monitor weather conditions for disease outbreaks"
        ]
      }
    };
  }

  async loadModel() {
    try {
      console.log('🤖 Checking Detectron2 model availability...');
      
      // Test connection to Detectron2 service
      const response = await fetch(`${this.modelEndpoint.replace('/predict', '/health')}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('✅ Detectron2 model connected:', data);
        this.isModelLoaded = true;
        this.modelVersion = data.version || '1.0.0';
        return true;
      } else {
        throw new Error('Model service unavailable');
      }
    } catch (error) {
      console.warn('⚠️ Detectron2 not available, loading fallback:', error.message);
      
      // Import fallback service dynamically
      try {
        const { default: SimpleDiseaseDetectionService } = await import('./SimpleDiseaseDetectionService');
        this.fallbackService = new SimpleDiseaseDetectionService();
        await this.fallbackService.loadModel();
        this.isModelLoaded = true;
        return true;
      } catch (fallbackError) {
        console.error('❌ Failed to load fallback service:', fallbackError);
        return false;
      }
    }
  }

  async preprocessImage(imageElement) {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      // Keep original dimensions for better detection
      const originalWidth = imageElement.naturalWidth || imageElement.width;
      const originalHeight = imageElement.naturalHeight || imageElement.height;
      
      // Resize to optimal size for your model (adjust based on training)
      const maxSize = 800; // Adjust based on your training resolution
      const scale = Math.min(maxSize / originalWidth, maxSize / originalHeight);
      
      canvas.width = Math.round(originalWidth * scale);
      canvas.height = Math.round(originalHeight * scale);
      
      // Draw the scaled image
      ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
      
      // Convert to base64
      const base64Data = canvas.toDataURL('image/jpeg', 0.9);
      resolve({
        base64: base64Data,
        width: canvas.width,
        height: canvas.height,
        originalWidth: originalWidth,
        originalHeight: originalHeight,
        scale: scale
      });
    });
  }

  async predictWithDetectron(imageData, includeVisualization = true) {
    try {
      console.log('🔬 Sending image to Detectron2 model...');
      
      const requestData = {
        image: imageData.base64.split(',')[1], // Remove data:image/jpeg;base64, prefix
        version: this.modelVersion,
        include_visualization: includeVisualization,
        metadata: {
          originalWidth: imageData.originalWidth,
          originalHeight: imageData.originalHeight,
          preprocessedWidth: imageData.width,
          preprocessedHeight: imageData.height,
          scale: imageData.scale
        }
      };

      const response = await fetch(this.modelEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      console.log('✅ Detectron2 prediction received:', result);

      return this.formatDetectronResult(result, imageData);
    } catch (error) {
      console.error('❌ Detectron2 prediction failed:', error);
      throw error;
    }
  }

  formatDetectronResult(detectronOutput, imageData) {
    // Expected Detectron2 output format:
    // {
    //   "predictions": [
    //     {
    //       "class_id": 1,
    //       "confidence": 0.85,
    //       "bbox": [x1, y1, x2, y2],
    //       "mask": [...] // Optional segmentation mask
    //     }
    //   ],
    //   "visualization": "base64_encoded_image_with_overlays",
    //   "image_info": {...}
    // }

    if (!detectronOutput.predictions || detectronOutput.predictions.length === 0) {
      // No detections = healthy
      const healthyResult = this.diseaseMapping[5];
      return {
        disease: healthyResult.name,
        confidence: 92.5,
        severity: healthyResult.severity,
        marathi: healthyResult.marathi,
        recommendations: healthyResult.recommendations,
        detectedRegions: 0,
        healthyArea: 100,
        visualizationImage: detectronOutput.visualization || null,
        detectionDetails: {
          boundingBoxes: [],
          masks: [],
          modelVersion: this.modelVersion,
          processingTime: detectronOutput.processing_time || 0,
          imageSize: `${imageData.width}x${imageData.height}`,
          originalSize: `${imageData.originalWidth}x${imageData.originalHeight}`
        }
      };
    }

    // Get highest confidence detection
    const bestDetection = detectronOutput.predictions.reduce((best, current) => 
      current.confidence > best.confidence ? current : best
    );

    const diseaseInfo = this.diseaseMapping[bestDetection.class_id] || this.diseaseMapping[5];
    const confidence = Math.round(bestDetection.confidence * 1000) / 10; // Convert to percentage

    // Calculate affected area based on bounding boxes
    const totalImageArea = imageData.width * imageData.height;
    let affectedArea = 0;
    
    detectronOutput.predictions.forEach(pred => {
      if (pred.bbox) {
        const [x1, y1, x2, y2] = pred.bbox;
        const bboxArea = Math.max(0, (x2 - x1) * (y2 - y1));
        affectedArea += bboxArea;
      }
    });

    const affectedPercentage = Math.min(100, (affectedArea / totalImageArea) * 100);
    const healthyArea = Math.max(0, 100 - affectedPercentage);

    // Count detections by disease type
    const diseaseCount = {};
    detectronOutput.predictions.forEach(pred => {
      const diseaseKey = pred.class_id;
      diseaseCount[diseaseKey] = (diseaseCount[diseaseKey] || 0) + 1;
    });

    return {
      disease: diseaseInfo.name,
      confidence: confidence,
      severity: diseaseInfo.severity,
      marathi: diseaseInfo.marathi,
      recommendations: diseaseInfo.recommendations,
      detectedRegions: detectronOutput.predictions.length,
      healthyArea: Math.round(healthyArea),
      visualizationImage: detectronOutput.visualization || null, // Base64 image with overlays
      detectionDetails: {
        boundingBoxes: detectronOutput.predictions.map(pred => ({
          class_id: pred.class_id,
          confidence: Math.round(pred.confidence * 1000) / 10,
          bbox: pred.bbox,
          class_name: this.diseaseMapping[pred.class_id]?.name || 'Unknown',
          area: pred.bbox ? (pred.bbox[2] - pred.bbox[0]) * (pred.bbox[3] - pred.bbox[1]) : 0
        })),
        masks: detectronOutput.predictions.map(pred => pred.mask).filter(mask => mask), // Segmentation masks if available
        diseaseBreakdown: diseaseCount,
        modelVersion: this.modelVersion,
        processingTime: detectronOutput.processing_time || 0,
        imageSize: `${imageData.width}x${imageData.height}`,
        originalSize: `${imageData.originalWidth}x${imageData.originalHeight}`,
        totalDetections: detectronOutput.predictions.length,
        affectedAreaPercentage: Math.round(affectedPercentage * 10) / 10,
        highConfidenceDetections: detectronOutput.predictions.filter(pred => pred.confidence > 0.7).length
      }
    };
  }

  async predict(imageElement, includeVisualization = true) {
    try {
      // Preprocess image
      const imageData = await this.preprocessImage(imageElement);
      
      // Try Detectron2 first
      if (this.isModelLoaded && !this.fallbackService) {
        try {
          return await this.predictWithDetectron(imageData, includeVisualization);
        } catch (detectronError) {
          console.warn('⚠️ Detectron2 failed, switching to fallback:', detectronError.message);
          
          // Load fallback service on demand
          if (!this.fallbackService) {
            const { default: SimpleDiseaseDetectionService } = await import('./SimpleDiseaseDetectionService');
            this.fallbackService = new SimpleDiseaseDetectionService();
            await this.fallbackService.loadModel();
          }
          
          const fallbackResult = await this.fallbackService.predict(imageElement);
          // Add empty visualization for fallback
          fallbackResult.visualizationImage = null;
          fallbackResult.detectionDetails = {
            ...fallbackResult.detectionDetails,
            fallbackMode: true,
            note: "Using color-based analysis (Detectron2 unavailable)"
          };
          return fallbackResult;
        }
      } else if (this.fallbackService) {
        // Use fallback service
        console.log('🔄 Using fallback detection service...');
        const fallbackResult = await this.fallbackService.predict(imageElement);
        fallbackResult.visualizationImage = null;
        fallbackResult.detectionDetails = {
          ...fallbackResult.detectionDetails,
          fallbackMode: true
        };
        return fallbackResult;
      } else {
        throw new Error('No detection service available');
      }
    } catch (error) {
      console.error('❌ Disease prediction failed:', error);
      
      // Last resort: return error result
      return {
        disease: "Detection Error",
        confidence: 0,
        severity: "Unknown",
        marathi: "तपासणी त्रुटी",
        recommendations: [
          "Please try again with a clearer image",
          "Ensure good lighting when taking photos",
          "Check your internet connection",
          "Contact support if problem persists"
        ],
        detectedRegions: 0,
        healthyArea: 0,
        visualizationImage: null,
        detectionDetails: {
          error: error.message,
          timestamp: new Date().toISOString()
        }
      };
    }
  }

  // Additional utility methods
  isDetectronAvailable() {
    return this.isModelLoaded && !this.fallbackService;
  }

  getModelInfo() {
    return {
      type: this.fallbackService ? 'Fallback (Color Analysis)' : 'Detectron2 Instance Segmentation',
      version: this.modelVersion,
      endpoint: this.modelEndpoint,
      status: this.isModelLoaded ? 'Ready' : 'Not Loaded',
      classes: Object.values(this.diseaseMapping).map(d => d.name),
      capabilities: this.fallbackService ? ['Color Analysis'] : ['Object Detection', 'Instance Segmentation', 'Visualization']
    };
  }

  // Get disease information by ID
  getDiseaseInfo(classId) {
    return this.diseaseMapping[classId] || null;
  }

  // Get all supported diseases
  getAllDiseases() {
    return this.diseaseMapping;
  }
}

export default DetectronDiseaseService;